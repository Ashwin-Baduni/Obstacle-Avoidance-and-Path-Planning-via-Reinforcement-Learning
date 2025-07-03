# Reinforcement Learning Algorithms Suite

A comprehensive implementation and comparison of three fundamental reinforcement learning algorithms: Q-Learning, Deep Q-Network (DQN), and Actor-Critic, tested on a GridWorld environment.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project provides a clean, modular implementation of three core reinforcement learning algorithms, allowing for easy comparison and analysis of their performance on a standardized GridWorld environment. Each algorithm is implemented with modern Python practices, comprehensive logging, and detailed visualization capabilities.

### Implemented Algorithms

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| **Q-Learning** | Tabular | Classic temporal difference learning, Q-table visualization |
| **Deep Q-Network (DQN)** | Deep RL | Neural network approximation, experience replay, ε-greedy exploration |
| **Actor-Critic** | Policy Gradient | Separate actor and critic networks, advantage-based learning |

## 🏗️ Project Architecture

```
├── main.py                    # Interactive main execution file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── algorithms/               # RL algorithm implementations
│   ├── __init__.py
│   ├── q_learning.py         # Q-Learning implementation
│   ├── dqn_agent.py          # Deep Q-Network implementation
│   └── actor_critic.py       # Actor-Critic implementation
├── core/                     # Core system components
│   ├── __init__.py
│   ├── environment.py        # GridWorld environment
│   └── utils.py             # Visualization and analysis utilities
└── outputs/                  # Generated results (created at runtime)
    ├── q_learning/           # Q-Learning specific outputs
    ├── dqn/                  # DQN specific outputs
    ├── actor_critic/         # Actor-Critic specific outputs
    └── comparisons/          # Cross-algorithm comparisons
```

## 🎮 GridWorld Environment

The testing environment is a **3×3 GridWorld** featuring:

- **Start Position**: (0,0) - Top-left corner
- **Goal Position**: (2,2) - Bottom-right corner  
- **Obstacle**: (1,1) - Center position (impassable)
- **Action Space**: 4 discrete actions (Up, Right, Down, Left)
- **Reward Structure**:
  - Goal reached: +100
  - Obstacle hit: -100
  - Each step: -1 (encourages efficiency)

```
┌─────┬─────┬─────┐
│  S  │     │     │  S = Start
├─────┼─────┼─────┤
│     │  ■  │     │  ■ = Obstacle
├─────┼─────┼─────┤
│     │     │  G  │  G = Goal
└─────┴─────┴─────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for accelerated training

### Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/Ashwin-Baduni/Reinforcement-Learning-Algorithms.git
   cd Reinforcement-Learning-Algorithms
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the interactive suite**:
   ```
   python main.py
   ```

### Usage Options

The main interface provides several execution modes:

```
1. Q-Learning          - Run tabular Q-Learning algorithm
2. Deep Q-Network      - Run DQN with neural network approximation  
3. Actor-Critic        - Run Actor-Critic policy gradient method
4. Run All Algorithms  - Execute all three with comprehensive comparison
5. Exit               - Terminate the program
```

## 📊 Features & Capabilities

### Training Features
- **Real-time Progress Tracking**: Live progress bars with performance metrics
- **Configurable Hyperparameters**: Easy modification of learning rates, exploration, etc.
- **GPU Acceleration**: Automatic CUDA utilization for TensorFlow operations
- **Organized Output Management**: Structured file organization for all results

### Analysis & Visualization
- **Performance Plots**: Training curves with raw scores and moving averages
- **Algorithm Comparison**: Side-by-side performance analysis
- **Detailed Reports**: Comprehensive statistical analysis of results
- **Data Export**: CSV files for external analysis tools

### Output Organization
All results are automatically organized into timestamped files:

```
outputs/
├── q_learning/
│   ├── logs/           # Detailed training logs
│   ├── visualizations/ # Performance plots  
│   └── data/          # Raw score data (CSV)
├── dqn/               # DQN-specific results
├── actor_critic/      # Actor-Critic results
└── comparisons/       # Cross-algorithm analysis
    ├── visualizations/ # Comparison plots
    └── reports/       # Performance summaries
```

## 🔧 Technical Implementation

### Q-Learning
- **Algorithm**: Temporal Difference Learning with Q-table
- **Exploration**: ε-greedy strategy
- **Episodes**: 1,000 training episodes
- **Features**: Complete Q-table visualization, optimal policy display

### Deep Q-Network (DQN)
- **Network Architecture**: 2 hidden layers (24 neurons each)
- **Experience Replay**: 2,000-step memory buffer
- **Exploration**: Decaying ε-greedy (1.0 → 0.01)
- **Episodes**: 500 training episodes
- **Input Encoding**: One-hot state representation

### Actor-Critic
- **Architecture**: Separate actor and critic networks
- **Actor**: Softmax policy output (action probabilities)
- **Critic**: Value function approximation
- **Learning**: Advantage-based policy gradients
- **Episodes**: 500 training episodes

## 📈 Performance Metrics

The suite tracks and reports multiple performance indicators:

- **Episode Scores**: Cumulative reward per episode
- **Learning Speed**: Episodes required to reach 90% performance
- **Stability**: Performance consistency (standard deviation)
- **Final Performance**: Average of last 100 episodes
- **Convergence Analysis**: Moving average trends

## 🛠️ Customization

### Environment Modification
Modify `core/environment.py` to:
- Change grid size
- Adjust reward structure  
- Add multiple goals or obstacles
- Implement different action spaces

### Algorithm Tuning
Each algorithm file contains configurable hyperparameters:
- Learning rates
- Exploration parameters
- Network architectures
- Training episodes

### Visualization Enhancement
Extend `core/utils.py` to add:
- Custom plot styles
- Additional metrics
- Export formats
- Interactive visualizations

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional RL algorithms (PPO, A3C, etc.)
- More complex environments
- Hyperparameter optimization
- Performance benchmarking
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python practices and clean architecture principles[1]
- Implements state-of-the-art reinforcement learning techniques[2]
- Uses professional package management and dependency organization[3]

## 📞 Contact

**Ashwin Baduni**
- GitHub: [@Ashwin-Baduni](https://github.com/Ashwin-Baduni)
- Email: baduniashwin@gmail.com

---
