# Railway Optimization System

A comprehensive solution for railway freight network optimization, focusing on efficient train routing, yard management, and service planning.

## Project Overview

This repository contains a mathematical optimization framework designed to minimize operational costs in railway freight networks while satisfying capacity and routing constraints. The system handles:

- Network data processing and transformation
- Visualization of railway networks and demand flows
- Calculation of shortest paths using Dijkstra's algorithm
- Mathematical model formulation with multiple objective components:
  - $\sum_{i} \sum_{j} c_{ij}y_{ij}$ - Train operation costs
  - $\sum_{s} \sum_{i} \sum_{j} d_{ij}x^s_{ij}$ - Distance-driven shipment costs
  - $\sum_{j} C_{yard,j}f_{ij}$ - Yard handling costs
  - $\sum_{k} P_{shifted,k}S_{k}$ - Penalties for shifted service cars

## Key Components

### 1. DataProcessor Class

The `DataProcessor` class (in `Railway Optimization DataProcessor Implementation.py`) processes raw network data and transforms it into the standardized format required for optimization. Key functionality includes:

- Loading network data from multiple sources
- Normalizing flow variables for heterogeneous freight units
- Calculating shortest paths using Dijkstra's algorithm
- Creating adjacency matrices for network topology
- Visualizing the railway network and demand flows
- Generating synthetic data for testing

### 2. Mathematical Optimization Model

Our optimization model (detailed in the accompanying documentation) focuses on:

#### Objective Function
Minimizes the total operational cost with four main components:
- Train operation costs
- Distance-driven shipment costs
- Yard handling costs
- Penalties for shifted service cars

#### Constraints
- Flow conservation constraints
- Train capacity constraints
- Yard capacity constraints (for normal and shifted classification)
- Routing constraints for shifted cars

## Quick Start

```python
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
```

## Key Features

- **Network Visualization**: Generate intuitive visualizations of railway networks, including nodes, edges, and demand flows.
- **Efficient Data Processing**: Transform raw data into optimization-ready formats with normalization and standardization.
- **Flexible Model Formulation**: Adapt constraints and parameters to match specific operational scenarios.
- **Synthetic Data Generation**: Create test scenarios with various network topologies and demand patterns.
- **Mathematical Optimization**: Solve complex routing and yard management problems using mixed-integer linear programming techniques.

## Mathematical Formulation

Our model formulates the railway optimization problem as a mixed-integer linear program. The complete mathematical formulation is as follows:

$
\min \sum_{i} \sum_{j} c_{ij}y_{ij} + \sum_{s} \sum_{i} \sum_{j} d_{ij}x^s_{ij} + \sum_{j} C_{yard,j}f_{ij} + \sum_{k} P_{shifted,k}S_{k}
$

Subject to the following constraints:

1. **Flow Conservation Constraints**:
   
   $\sum_{j} x^s_{ij} = 1, \forall s \in S, i, j \in N$
   
   $\sum_{j} x^s_{ij} = \sum_{k} x^s_{jk}, \forall j, s$

2. **Train Capacity Constraints**:
   
   $x^s_{ij} \leq y_{ij}, \forall i, j, s$
   
   $\sum_{s} x^s_{ij}n_s \leq L_{Max}, \forall i, j$

3. **Yard Capacity Constraints**:
   
   $\sum_{j} f_{ij} \leq C_{Yard,i}, \forall i$
   
   $\sum_{j} x^s_{ij} \leq C_{Sort,i}, \forall i$
   
   For double-hump yards:
   
   $\sum_{j} x^s_{ij}h^0_{ijk} \leq C_{Normal,k}, \forall k$
   
   $\sum_{j} x^s_{ij}h^1_{ijk} \leq C_{Shifted,k}, \forall k$

4. **Routing Constraints with Shifted Cars**:
   
   $h^0_{ijk} + h^1_{ijk} \leq 1, \forall i, j, k$
   
   $h^1_{ijk} \leq y_{ij}, \forall i, j, k$
   
The model distinguishes between normal and shifted classification operations at yards, introducing penalties to discourage inefficient routing.

## Recent Simplifications

Our latest update streamlines the mathematical formulation while preserving core functionality:

1. **Constraint Reduction**: Reduced redundant constraints for individual yard conditions by implementing a unified set of constraints applicable to all classification yards
2. **Classification Simplification**: Merged classification yard constraints by distinguishing only between normal ($h^0_{ijk}$) and shifted ($h^1_{ijk}$) classifications rather than modeling multiple levels of routing restrictions
3. **Variable Type Optimization**: Eliminated unnecessary integer constraints where binary indicators ($x^s_{ij}$, $y_{ij}$, etc.) suffice to enforce logical conditions
4. **Penalty Approach**: Added penalty terms for shifted cars ($P_{shifted,k}S_{k}$) instead of modeling complex yard congestion effects explicitly

These changes make the model more computationally efficient while still capturing essential trade-offs in freight train service planning.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- NetworkX
- Matplotlib
- (Optional) Mathematical optimization solver


## Acknowledgements

This work was developed by Group 17:
- F. Feng
- S. Xiong
- Z. Fan
- E. Qu

February 2025
