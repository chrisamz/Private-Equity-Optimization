pip install torch torch-geometric numpy pandas

import torch
import numpy as np
from torch_geometric.data import Data
import pandas as pd
import networkx as nx

# Example financial metrics and market data for companies
companies = {
    'AAPL': {'revenue': 274515000000, 'profit': 57411000000, 'assets': 323888000000, 'liabilities': 258549000000},
    'MSFT': {'revenue': 143015000000, 'profit': 44281000000, 'assets': 301311000000, 'liabilities': 183007000000},
    'GOOGL': {'revenue': 182527000000, 'profit': 40269000000, 'assets': 319616000000, 'liabilities': 97084000000},
    'AMZN': {'revenue': 386064000000, 'profit': 21331000000, 'assets': 321195000000, 'liabilities': 227791000000}
}

# Example edge data representing relationships (e.g., industry similarity, partnerships)
edges = [
    ('AAPL', 'MSFT', {'weight': 0.9}),
    ('AAPL', 'GOOGL', {'weight': 0.8}),
    ('MSFT', 'GOOGL', {'weight': 0.85}),
    ('MSFT', 'AMZN', {'weight': 0.7}),
    ('GOOGL', 'AMZN', {'weight': 0.75}),
]

# Convert the companies dictionary to a DataFrame
df_companies = pd.DataFrame.from_dict(companies, orient='index')

# Normalize the financial metrics
df_companies = (df_companies - df_companies.mean()) / df_companies.std()

# Convert the DataFrame to a tensor
node_features = torch.tensor(df_companies.values, dtype=torch.float)

# Create a NetworkX graph
G = nx.Graph()

# Add nodes with features
for i, company in enumerate(df_companies.index):
    G.add_node(i, feature=node_features[i], name=company)

# Add edges with weights
for edge in edges:
    node1 = df_companies.index.get_loc(edge[0])
    node2 = df_companies.index.get_loc(edge[1])
    G.add_edge(node1, node2, weight=edge[2]['weight'])

# Convert the NetworkX graph to PyTorch Geometric data
edge_index = torch.tensor(list(G.edges)).t().contiguous()
edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float).view(-1, 1)
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class InvestmentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(InvestmentGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# Example usage
in_channels = node_features.shape[1]
hidden_channels = 32
out_channels = 1  # Predicting potential growth or valuation
model = InvestmentGNN(in_channels, hidden_channels, out_channels)

from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader

# Example labels for the companies (e.g., potential growth or valuation)
labels = torch.tensor([0.8, 0.7, 0.9, 0.85], dtype=torch.float)

# Split data into train and test sets
train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42)
train_data = data[train_idx]
test_data = data[test_idx]

train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
test_loader = DataLoader([test_data], batch_size=1, shuffle=False)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training loop
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out.view(-1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation loop
def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out.view(-1), data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

# Train and evaluate the model
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    train_loss = train()
    test_loss = evaluate(test_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

