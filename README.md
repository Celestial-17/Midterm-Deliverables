Railway Optimization System
A comprehensive solution for railway freight network optimization, focusing on efficient train routing, yard management, and service planning.
Project Overview
This repository contains a mathematical optimization framework designed to minimize operational costs in railway freight networks while satisfying capacity and routing constraints. The system handles:
Network data processing and transformation
Visualization of railway networks and demand flows
Calculation of shortest paths and baseline costs
Mathematical model for cost minimization with multiple constraints
Key Components
1. DataProcessor Class
The DataProcessor class (in Railway Optimization DataProcessor Implementation.py) processes raw network data and transforms it into the standardized format required for optimization. Key functionality includes:
Loading network data from multiple sources
Normalizing flow variables for heterogeneous freight units
Calculating shortest paths using Dijkstra's algorithm
Creating adjacency matrices for network topology
Visualizing the railway network and demand flows
Generating synthetic data for testing
2. Mathematical Optimization Model
Our optimization model (detailed in the accompanying documentation) focuses on:
Objective Function
Minimizes the total operational cost with four main components:
Train operation costs
Distance-driven shipment costs
Yard handling costs
Penalties for shifted service cars
Constraints
Flow conservation constraints
Train capacity constraints
Yard capacity constraints (for normal and shifted classification)
Routing constraints for shifted cars
Quick Start
# Import necessary library
from data_processor import DataProcessor

# Create processor instance
processor = DataProcessor()

# Option 1: Generate synthetic data for testing
processor.generate_synthetic_data(num_nodes=15, seed=42)

# Option 2: Load your own network data
# processor.load_network_data(nodes_data, edges_data, demands_data, yard_capacities_data)

# Visualize the network
processor.visualize_network(show_demands=True)

# Calculate and print baseline costs
baseline_costs = processor.get_baseline_costs()
print(f"Total baseline routing cost: {sum(baseline_costs.values()):.2f}")

# Export data for optimization
optimization_data = processor.export_for_optimization()
Key Features
Network Visualization: Generate intuitive visualizations of railway networks, including nodes, edges, and demand flows.
Efficient Data Processing: Transform raw data into optimization-ready formats with normalization and standardization.
Flexible Model Formulation: Adapt constraints and parameters to match specific operational scenarios.
Synthetic Data Generation: Create test scenarios with various network topologies and demand patterns.
Mathematical Formulation
The optimization model minimizes:
min∑∑cᵢⱼyᵢⱼ + ∑∑∑dᵢⱼxᵢⱼˢ + ∑Cyard,ⱼfᵢⱼ + ∑Pshifted,ₖSₖ
   i j        s i j          j              k
Subject to flow conservation, capacity, and routing constraints. The model distinguishes between normal and shifted classification operations at yards, introducing penalties to discourage inefficient routing.
Recent Simplifications
Our latest update streamlines the mathematical formulation while preserving core functionality:
Reduced redundant constraints for individual yard conditions
Merged classification yard constraints by distinguishing only between normal and shifted classifications
Eliminated unnecessary integer constraints where binary indicators suffice
Added penalty terms for shifted cars instead of modeling complex yard congestion effects
These changes make the model more computationally efficient while still capturing essential trade-offs in freight train service planning.
Requirements
Python 3.7+
NumPy
Pandas
NetworkX
Matplotlib
(Optional) Mathematical optimization solver
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
Acknowledgements
This work was developed by Group 17:
F. Feng
S. Xiong
Z. Fan
E. Qu
February 2025
