import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt


class DataProcessor:
"""
DataProcessor for railway optimization model that transforms raw network data 
into standardized format required for optimization.

This class implements key mathematical transformations:
1. Normalization of flow variables for heterogeneous freight units
2. Calculation of shortest paths using Dijkstra's algorithm
3. Construction of adjacency matrices for network topology
"""

def __init__(self):
"""Initialize the DataProcessor with empty data structures."""
self.raw_network = None
self.nodes = []
self.edges = []
self.demands = {}
self.distances = None
self.yard_capacities = {}
self.train_capacities = {}
self.unit_costs = {}
self.graph = None
self.shortest_paths = {}
self.adjacency_matrix = None

def load_network_data(self, 
nodes_data: pd.DataFrame, 
edges_data: pd.DataFrame, 
demands_data: pd.DataFrame,
yard_capacities_data: Optional[pd.DataFrame] = None,
train_capacities_data: Optional[pd.DataFrame] = None) -> None:
"""
Load raw network data from DataFrames.

Parameters:
-----------
nodes_data: DataFrame with columns ['node_id', 'name', 'type', 'x_coord', 'y_coord']
edges_data: DataFrame with columns ['from_node', 'to_node', 'distance', 'capacity']
demands_data: DataFrame with columns ['origin', 'destination', 'volume']
yard_capacities_data: DataFrame with columns ['node_id', 'capacity']
train_capacities_data: DataFrame with columns ['train_id', 'capacity']
"""
self.nodes = nodes_data
self.edges = edges_data

# Create NetworkX graph from data
self.graph = nx.DiGraph()

# Add nodes
for _, node in nodes_data.iterrows():
self.graph.add_node(node['node_id'], 
name=node['name'], 
type=node['type'],
x_coord=node.get('x_coord', 0),
y_coord=node.get('y_coord', 0))

# Add edges with distances as weights
for _, edge in edges_data.iterrows():
self.graph.add_edge(edge['from_node'], 
edge['to_node'], 
distance=edge['distance'],
capacity=edge.get('capacity', float('inf')))

# Process demands into a dictionary format
self.demands = {(row['origin'], row['destination']): row['volume'] 
for _, row in demands_data.iterrows()}

# Process yard capacities if provided
if yard_capacities_data is not None:
self.yard_capacities = {row['node_id']: row['capacity'] 
for _, row in yard_capacities_data.iterrows()}

# Process train capacities if provided
if train_capacities_data is not None:
self.train_capacities = {row['train_id']: row['capacity'] 
for _, row in train_capacities_data.iterrows()}

# Calculate all-pairs shortest paths
self._calculate_shortest_paths()

# Create adjacency matrix
self._create_adjacency_matrix()

def normalize_flow_variables(self, 
flow_data: pd.DataFrame, 
normalization_type: str = 'minmax') -> pd.DataFrame:
"""
Normalize flow variables to account for heterogeneous freight units.

Parameters:
-----------
flow_data: DataFrame with flow variables
normalization_type: Type of normalization ('minmax' or 'zscore')

Returns:
--------
Normalized DataFrame
"""
normalized_data = flow_data.copy()

if normalization_type == 'minmax':
# Min-max normalization to [0, 1] range
for col in normalized_data.select_dtypes(include=[np.number]).columns:
min_val = normalized_data[col].min()
max_val = normalized_data[col].max()
if max_val > min_val:
normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)

elif normalization_type == 'zscore':
# Z-score normalization (standardization)
for col in normalized_data.select_dtypes(include=[np.number]).columns:
mean_val = normalized_data[col].mean()
std_val = normalized_data[col].std()
if std_val > 0:
normalized_data[col] = (normalized_data[col] - mean_val) / std_val

return normalized_data

def _calculate_shortest_paths(self) -> None:
"""
Calculate shortest paths using Dijkstra's algorithm for all pairs of nodes.
Stores results in self.shortest_paths.
"""
# Get all shortest paths using NetworkX
self.shortest_paths = {}

# Extract distances from graph
for source in self.graph.nodes():
self.shortest_paths[source] = {}
# Calculate shortest paths from this source to all destinations
path_lengths = nx.single_source_dijkstra_path_length(
self.graph, source, weight='distance')

for target, length in path_lengths.items():
self.shortest_paths[source][target] = length

def get_shortest_path(self, 
origin: int, 
destination: int, 
return_path: bool = False) -> Union[float, Tuple[float, List]]:
"""
Get the shortest path distance and optionally the path itself.

Parameters:
-----------
origin: Origin node ID
destination: Destination node ID
return_path: Whether to return the path nodes (default: False)

Returns:
--------
If return_path is False: shortest path distance (float)
If return_path is True: tuple of (distance, path)
"""
if origin not in self.graph or destination not in self.graph:
raise ValueError(f"Nodes {origin} and/or {destination} not in graph")

if return_path:
length, path = nx.single_source_dijkstra(
self.graph, origin, destination, weight='distance')
return length, path
else:
return self.shortest_paths[origin][destination]

def _create_adjacency_matrix(self) -> None:
"""
Create adjacency matrix representation of the network.
"""
# Get sorted list of nodes for consistent indexing
nodes_list = sorted(list(self.graph.nodes()))
n = len(nodes_list)

# Create mapping from node id to matrix index
node_to_idx = {node: i for i, node in enumerate(nodes_list)}

# Initialize adjacency matrix with infinity
self.adjacency_matrix = np.full((n, n), np.inf)

# Set diagonal to zero (distance from node to itself)
np.fill_diagonal(self.adjacency_matrix, 0)

# Fill in edge weights
for u, v, data in self.graph.edges(data=True):
i, j = node_to_idx[u], node_to_idx[v]
self.adjacency_matrix[i, j] = data['distance']

def get_baseline_costs(self) -> Dict:
"""
Calculate baseline routing costs for each origin-destination pair.

Returns:
--------
Dictionary with (origin, destination) tuples as keys and costs as values
"""
routing_costs = {}

for (origin, destination), volume in self.demands.items():
distance = self.get_shortest_path(origin, destination)

# Calculate cost (could be customized based on project requirements)
# Here we use a simple distance-based cost
cost = distance * volume

routing_costs[(origin, destination)] = cost

return routing_costs

def visualize_network(self, 
figsize: Tuple[int, int] = (12, 10), 
with_labels: bool = True,
show_demands: bool = False) -> None:
"""
Visualize the railway network.

Parameters:
-----------
figsize: Figure size
with_labels: Whether to show node labels
show_demands: Whether to show demand flows
"""
plt.figure(figsize=figsize)

# Get positions from node attributes or use spring layout
if all('x_coord' in data and 'y_coord' in data for _, data in self.graph.nodes(data=True)):
pos = {node: (data['x_coord'], data['y_coord']) 
for node, data in self.graph.nodes(data=True)}
else:
pos = nx.spring_layout(self.graph)

# Draw nodes and edges
nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color='lightblue')
nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.7, arrows=True)

if with_labels:
nx.draw_networkx_labels(self.graph, pos)

# Draw edge labels (distances)
edge_labels = {(u, v): f"{d['distance']:.1f}" 
for u, v, d in self.graph.edges(data=True)}
nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

# Optionally show demands as curved edges
if show_demands and self.demands:
demand_graph = nx.DiGraph()
for (origin, destination), volume in self.demands.items():
if origin != destination: # Skip self-loops
demand_graph.add_edge(origin, destination, weight=volume)

# Draw curved demand edges in red
curved_edges = [edge for edge in demand_graph.edges() 
if edge not in self.graph.edges()]
nx.draw_networkx_edges(demand_graph, pos, edgelist=curved_edges, 
edge_color='red', style='dashed', 
connectionstyle='arc3,rad=0.3')

plt.title("Railway Network Visualization")
plt.axis('off')
plt.tight_layout()
plt.show()

def export_for_optimization(self) -> Dict:
"""
Export processed data in format ready for optimization model.

Returns:
--------
Dictionary with all necessary data structures for optimization
"""
# Prepare data in format needed for optimization
export_data = {
'nodes': list(self.graph.nodes()),
'edges': [(u, v) for u, v in self.graph.edges()],
'distances': {(u, v): data['distance'] for u, v, data in self.graph.edges(data=True)},
'demands': self.demands,
'yard_capacities': self.yard_capacities,
'train_capacities': self.train_capacities,
'shortest_paths': self.shortest_paths,
'adjacency_matrix': self.adjacency_matrix.tolist()
}

return export_data

def generate_synthetic_data(self, 
num_nodes: int = 10, 
edge_probability: float = 0.3,
demand_probability: float = 0.2,
seed: Optional[int] = None) -> None:
"""
Generate synthetic network data for testing.

Parameters:
-----------
num_nodes: Number of nodes in the network
edge_probability: Probability of edge between any two nodes
demand_probability: Probability of demand between any two nodes
seed: Random seed for reproducibility
"""
if seed is not None:
np.random.seed(seed)

# Generate random graph
random_graph = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability, directed=True, seed=seed)

# Ensure connectivity by adding edges if needed
components = list(nx.weakly_connected_components(random_graph))
if len(components) > 1:
# Connect components
for i in range(len(components) - 1):
u = np.random.choice(list(components[i]))
v = np.random.choice(list(components[i+1]))
random_graph.add_edge(u, v)

# Assign random distances to edges
for u, v in random_graph.edges():
random_graph[u][v]['distance'] = np.random.uniform(10, 100)
random_graph[u][v]['capacity'] = np.random.uniform(50, 200)

# Create nodes DataFrame
nodes_data = pd.DataFrame({
'node_id': list(random_graph.nodes()),
'name': [f'Node_{i}' for i in random_graph.nodes()],
'type': np.random.choice(['yard', 'station', 'junction'], size=len(random_graph.nodes())),
'x_coord': np.random.uniform(0, 100, size=len(random_graph.nodes())),
'y_coord': np.random.uniform(0, 100, size=len(random_graph.nodes()))
})

# Create edges DataFrame
edges_list = []
for u, v, data in random_graph.edges(data=True):
edges_list.append({
'from_node': u,
'to_node': v,
'distance': data['distance'],
'capacity': data['capacity']
})
edges_data = pd.DataFrame(edges_list)

# Generate random demands
demands_list = []
for u in random_graph.nodes():
for v in random_graph.nodes():
if u != v and np.random.random() < demand_probability:
demands_list.append({
'origin': u,
'destination': v,
'volume': np.random.uniform(10, 50)
})
demands_data = pd.DataFrame(demands_list)

# Generate yard capacities
yard_capacities_list = []
for i, row in nodes_data.iterrows():
if row['type'] == 'yard':
yard_capacities_list.append({
'node_id': row['node_id'],
'capacity': np.random.uniform(100, 500)
})
yard_capacities_data = pd.DataFrame(yard_capacities_list)

# Load the generated data
self.load_network_data(nodes_data, edges_data, demands_data, yard_capacities_data)


# If you want to try
if __name__ == "__main__":
# Create processor instance
processor = DataProcessor()

# Generate synthetic data for testing
processor.generate_synthetic_data(num_nodes=15, seed=42)

# Visualize the network
processor.visualize_network(show_demands=True)

# Calculate and print baseline costs
baseline_costs = processor.get_baseline_costs()
print(f"Total baseline routing cost: {sum(baseline_costs.values()):.2f}")

# Export data for optimization
optimization_data = processor.export_for_optimization()
print(f"Prepared {len(optimization_data['nodes'])} nodes and {len(optimization_data['edges'])} edges for optimization")
print(f"Number of demand pairs: {len(optimization_data['demands'])}")