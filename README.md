# Dynamic Load Balancing in Multiprocessor Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An educational simulation demonstrating dynamic load balancing algorithms in multiprocessor systems. This project visualizes how operating systems distribute workloads across multiple CPUs to optimize performance and resource utilization.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Load Balancing Algorithms](#-load-balancing-algorithms)
- [Project Structure](#-project-structure)
- [Performance Metrics](#-performance-metrics)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

In modern computing systems, multiple processors (CPUs/cores) work together to execute tasks. **Load balancing** is the process of distributing workloads across these processors to:

- **Maximize throughput** - Complete more work in less time
- **Minimize response time** - Users get faster responses
- **Optimize resource utilization** - All processors stay busy
- **Prevent bottlenecks** - No single processor gets overwhelmed

This simulator allows you to visualize and compare different load balancing strategies in real-time.

## ✨ Features

### Core Features
- **Multi-Processor Simulation** - Simulate 4-8 virtual processors
- **Process Generation** - Create processes with random or custom attributes
- **Real-Time Visualization** - Watch processes being distributed and executed
- **Multiple Algorithms** - Compare Round Robin, Least Loaded, and Threshold-Based strategies
- **Process Migration** - Move processes between processors for better balance

### Visualization
- **Processor Load Bars** - Real-time load visualization with color coding
- **Gantt Chart** - Process execution timeline
- **Performance Dashboard** - Live metrics display
- **Comparison Charts** - Algorithm performance comparison

### Analysis
- **CPU Utilization** - Per-processor and average utilization
- **Waiting Time** - Average and per-process waiting times
- **Turnaround Time** - Total time from arrival to completion
- **Load Variance** - Measure of load distribution quality
- **Migration Statistics** - Track process movements

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Tkinter (usually included with Python)
- Matplotlib

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/dynamic_load_balancer.git
cd dynamic_load_balancer
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python main.py
```

## 📖 Usage

### GUI Mode (Default)
```bash
python main.py
```

This opens the graphical interface where you can:
1. Configure number of processors (4-8)
2. Set number of processes to simulate
3. Choose load balancing algorithm
4. Start/Stop/Reset simulation
5. View real-time visualization and metrics

### CLI Mode (Testing)
```bash
python main.py --cli
```

Runs a quick simulation in the terminal for testing purposes.

### Test Mode
```bash
python main.py --test
```

Runs module tests to verify all components work correctly.

### Command Line Options
```bash
python main.py --help

Options:
  --cli              Run in command-line mode (no GUI)
  --test             Run module tests
  --processors, -p   Number of processors (default: 4)
  --processes, -n    Number of processes (default: 20)
  --algorithm, -a    Load balancing algorithm
  --version, -v      Show version information
```

## ⚖️ Load Balancing Algorithms

### 1. Round Robin
**How it works:** Distributes processes to processors in a cyclic manner (P1→P2→P3→P4→P1→...).

**Pros:**
- Simple and fair
- Equal distribution by count
- Low overhead

**Cons:**
- Doesn't consider actual processor load
- Can lead to imbalance with varied process sizes

**Best for:** Homogeneous workloads with similar process sizes

### 2. Least Loaded First
**How it works:** Assigns each new process to the processor with the lowest current load.

**Pros:**
- Better load distribution
- Adapts to current system state
- Efficient for varied workloads

**Cons:**
- Slightly higher overhead
- Requires load monitoring

**Best for:** Variable workloads with different burst times

### 3. Threshold-Based
**How it works:** Monitors processor loads and migrates processes when the load difference exceeds a threshold.

**Pros:**
- Dynamic rebalancing
- Handles changing workloads
- Prevents severe imbalances

**Cons:**
- Migration has overhead cost
- Requires careful threshold tuning

**Best for:** Dynamic workloads where load changes over time

## 📁 Project Structure

```
dynamic_load_balancer/
├── main.py              # Application entry point
├── config.py            # Configuration constants and parameters
├── process.py           # Process class and generator
├── processor.py         # Processor class and manager
├── load_balancer.py     # Load balancing algorithms (Phase 3)
├── gui.py               # Graphical user interface (Phase 4)
├── metrics.py           # Performance calculations (Phase 5)
├── utils.py             # Helper functions and utilities
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── project.xml          # Project specification
```

### Module Descriptions

| Module | Description |
|--------|-------------|
| `config.py` | Configuration classes, enums, and constants |
| `process.py` | Process class with state management and lifecycle |
| `processor.py` | Processor simulation with queue and execution logic |
| `load_balancer.py` | Implementation of balancing algorithms |
| `gui.py` | Tkinter-based graphical interface |
| `metrics.py` | Performance metric calculations |
| `utils.py` | Logging, statistics, export utilities |

## 📊 Performance Metrics

### Process Metrics
| Metric | Formula | Description |
|--------|---------|-------------|
| **Turnaround Time** | Completion - Arrival | Total time in system |
| **Waiting Time** | Start - Arrival | Time spent waiting |
| **Response Time** | First Run - Arrival | Time until first execution |

### Processor Metrics
| Metric | Formula | Description |
|--------|---------|-------------|
| **CPU Utilization** | Execution / Total Time | Percentage of time busy |
| **Queue Length** | Count of waiting processes | Current workload |
| **Load** | Queue + Remaining Work | Combined load measure |

### System Metrics
| Metric | Description |
|--------|-------------|
| **Load Variance** | Standard deviation of processor loads |
| **Load Balance Index** | 1 - (max-min)/max, higher is better |
| **Jain's Fairness Index** | Statistical fairness measure (0 to 1) |
| **Total Migrations** | Number of process movements |

## 📸 Screenshots

*Screenshots will be added after GUI implementation in Phase 4*

## 🔧 Development

### Running Tests
```bash
python main.py --test
```

### Individual Module Tests
```bash
python config.py
python process.py
python processor.py
python utils.py
```

### Code Style
This project follows PEP 8 guidelines. All code includes:
- Comprehensive docstrings
- Type hints
- Inline comments for complex logic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Message Format
```
type: Brief description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- test: Adding tests
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Operating Systems concepts from Silberschatz, Galvin, and Gagne
- Python documentation and community
- Tkinter and Matplotlib libraries

---

**Made with ❤️ for learning Operating Systems concepts**
