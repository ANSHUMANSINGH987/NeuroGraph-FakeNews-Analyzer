import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ---------------------------------------------------------
# 1. NEUROGRAPH ARCHITECTURE
# ---------------------------------------------------------
class NeuroGraph(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(NeuroGraph, self).__init__()
        self.gat1 = GATConv(num_node_features, hidden_channels, heads=2)
        self.gat2 = GATConv(hidden_channels * 2, hidden_channels, heads=1)
        self.classifier = torch.nn.Linear(hidden_channels + 1, num_classes) 

    def forward(self, x, edge_index, batch, ipr_score):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        
        graph_embedding = global_max_pool(x, batch) 
        ipr_tensor = ipr_score.view(-1, 1).to(graph_embedding.device)
        joint_embedding = torch.cat([graph_embedding, ipr_tensor], dim=1)
        
        return F.log_softmax(self.classifier(joint_embedding), dim=1), joint_embedding

# ---------------------------------------------------------
# 2. LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    bot_model = joblib.load('models/bot_expert_model.pkl')
    feature_names = joblib.load('models/bot_feature_names.pkl')
    
    device = torch.device('cpu') 
    num_features = len(feature_names)
    gnn_model = NeuroGraph(num_node_features=num_features, hidden_channels=32, num_classes=2)
    gnn_model.load_state_dict(torch.load('models/neurograph_model.pth', map_location=device, weights_only=True))
    gnn_model.eval()
    
    return bot_model, gnn_model, num_features

try:
    bot_model, gnn_model, num_features = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models. Error: {e}")

# ---------------------------------------------------------
# 3. GRAPH DRAWING HELPER
# ---------------------------------------------------------
def draw_graph(edge_index, title, topology_type, ipr_value):
    G = nx.Graph()
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    
    plt.figure(figsize=(7, 5))
    
    # Color nodes based on the IPR probability (visualizing the bots)
    node_colors = []
    for node in G.nodes():
        if topology_type == "Star" and node == 0:
            node_colors.append('#d62728') # Master Bot is Red
        else:
            # Randomly color nodes red based on the IPR slider percentage
            is_bot = np.random.rand() < ipr_value
            node_colors.append('#ff9896' if is_bot else '#1f77b4')
            
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=80, edge_color='#cccccc', width=1.0)
    plt.title(title)
    return plt

# ---------------------------------------------------------
# 4. INTERACTIVE UI DASHBOARD
# ---------------------------------------------------------
st.set_page_config(page_title="NeuroGraph Live Demo", layout="wide", page_icon="🧠")
st.title("🧠 NeuroGraph: Interactive Detection Dashboard")
st.markdown("Adjust the parameters on the left to simulate different misinformation cascades and test the GNN in real-time.")

if models_loaded:
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🎛️ Simulation Controls")
    
    topology_type = st.sidebar.radio(
        "1. Select Network Topology:", 
        ("Organic Tree (Human-like)", "Centralized Star (Bot Attack)")
    )
    is_star = "Star" in topology_type
    
    num_nodes = st.sidebar.slider("2. Network Size (Users involved):", min_value=10, max_value=150, value=40, step=10)
    
    default_ipr = 0.85 if is_star else 0.15
    user_ipr = st.sidebar.slider("3. Bot Density (IPR Score):", min_value=0.0, max_value=1.0, value=default_ipr, step=0.05)
    st.sidebar.caption("💡 *Slide this to test edge cases! What happens if a human tree network gets hijacked by 90% bots?*")

    st.write("---")
    
    # --- GRAPH GENERATION ---
    if is_star:
        sources = torch.arange(1, num_nodes)
        targets = torch.zeros(num_nodes - 1, dtype=torch.long)
    else:
        sources = torch.arange(1, num_nodes)
        targets = torch.randint(0, num_nodes - 1, (num_nodes - 1,))
        targets = torch.min(targets, sources - 1)
        
    edge_index = torch.stack([torch.cat([sources, targets]), torch.cat([targets, sources])], dim=0)
    ipr_score = torch.tensor([user_ipr], dtype=torch.float)
    x = torch.randn((num_nodes, num_features), dtype=torch.float)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader(f"Network Visualization: {num_nodes} Users")
        st.markdown(f"**Red Nodes** represent suspected bots based on the **{user_ipr*100:.0f}% IPR score**.")
        fig = draw_graph(edge_index, f"{'Star' if is_star else 'Tree'} Topology Visualization", "Star" if is_star else "Tree", user_ipr)
        st.pyplot(fig)
        
    with col2:
        st.subheader("GNN AI Analysis")
        st.info("The model analyzes both the structural shape and the bot density to make a prediction.")
        
        if st.button("🚀 Run PyTorch Inference Live", type="primary", use_container_width=True):
            with st.spinner("Passing graph through Attention Layers..."):
                with torch.no_grad():
                    out, joint_embed = gnn_model(x, edge_index, batch, ipr_score)
                    probabilities = torch.exp(out)[0]
                    pred = out.argmax(dim=1).item()
            
            st.write("### Final Classification:")
            if pred == 1:
                st.error(f"🚨 FAKE NEWS DETECTED!")
                st.progress(float(probabilities[1]))
                st.write(f"**Confidence:** {probabilities[1]*100:.2f}%")
                st.write("The model identified a highly coordinated structural anomaly typical of disinformation campaigns.")
            else:
                st.success(f"✅ REAL NEWS DETECTED.")
                st.progress(float(probabilities[0]))
                st.write(f"**Confidence:** {probabilities[0]*100:.2f}%")
                st.write("The network exhibits deep, cascading topologies indicating natural human-to-human sharing.")
                
            # Academic Proof Expander
            with st.expander("🔍 View Raw Mathematical Output (For Reviewers)"):
                st.code(f"""
# 1. Input Tensor Shapes
Node Features (X): {list(x.shape)}
Edge Index (A): {list(edge_index.shape)}
IPR Score: {ipr_score.item()}

# 2. GAT Aggregation
Graph Embedding Shape: [1, 32]
Joint Representation (Graph + IPR): {list(joint_embed.shape)}

# 3. Softmax Probabilities
Real News Probability: {probabilities[0]:.4f}
Fake News Probability: {probabilities[1]:.4f}
                """, language="python")