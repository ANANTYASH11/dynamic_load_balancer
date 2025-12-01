#!/usr/bin/env python3
"""
Dynamic Load Balancing in Multiprocessor Systems - Main Entry Point

This is the main entry point for the Dynamic Load Balancing Simulator.
It initializes all components, sets up the GUI, and starts the application.

The simulation demonstrates key Operating System concepts:
- Process Management and Scheduling
- Multiprocessor Systems
- Load Balancing Algorithms
- Performance Metrics and Analysis

Usage:
    python main.py              # Run with GUI
    python main.py --cli        # Run in CLI mode (for testing)
    python main.py --test       # Run module tests

Author: Student
Date: December 2024
"""

import sys
import os
import argparse
import logging
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project modules
from config import (
    SimulationConfig,
    GUIConfig,
    LoggingConfig,
    LoadBalancingAlgorithm,
    VERSION,
    APP_NAME,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_GUI_CONFIG,
    DEFAULT_LOGGING_CONFIG
)
from process import Process, ProcessGenerator
from processor import Processor, ProcessorManager
from utils import (
    SimulationLogger,
    setup_logging,
    DataExporter,
    calculate_mean,
    calculate_std_dev,
    calculate_load_balance_index,
    calculate_jains_fairness_index
)


def print_banner():
    """Print application banner."""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       DYNAMIC LOAD BALANCING IN MULTIPROCESSOR SYSTEMS               ║
║                        Simulator v{VERSION}                             ║
║                                                                      ║
║   An educational simulation demonstrating OS load balancing concepts ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_cli_simulation():
    """
    Run a command-line simulation for testing purposes.
    
    This provides a quick way to test the core simulation logic
    without the GUI.
    """
    print_banner()
    print("\n[CLI Mode - Testing Core Simulation]\n")
    
    # Initialize components
    config = SimulationConfig(
        num_processors=4,
        num_processes=10,
        time_quantum=4
    )
    
    logger = setup_logging()
    logger.info("Starting CLI simulation")
    
    # Create processor manager
    processor_manager = ProcessorManager(
        num_processors=config.num_processors,
        config=config
    )
    
    # Generate processes
    generator = ProcessGenerator(config)
    processes = generator.generate_predefined_test_set()
    
    print(f"Configuration:")
    print(f"  Processors: {config.num_processors}")
    print(f"  Processes: {len(processes)}")
    print(f"  Time Quantum: {config.time_quantum}")
    print(f"  Algorithm: {config.default_algorithm.value}")
    
    print(f"\nGenerated Processes:")
    print("-" * 60)
    for p in processes:
        print(f"  P{p.pid}: arrival={p.arrival_time}, burst={p.burst_time}, "
              f"priority={p.priority.name}")
    
    # Simple Round Robin distribution for testing
    print(f"\n[Distributing processes using Round Robin]")
    rr_index = 0
    for p in processes:
        processor_id = rr_index % config.num_processors
        processor_manager[processor_id].add_process(p)
        print(f"  P{p.pid} -> Processor {processor_id}")
        rr_index += 1
    
    # Show initial loads
    print(f"\nInitial Processor Loads:")
    for proc in processor_manager:
        print(f"  Processor {proc.processor_id}: "
              f"queue={proc.get_queue_size()}, load={proc.get_load():.2f}")
    
    # Run simulation
    print(f"\n[Running Simulation]")
    print("-" * 60)
    
    current_time = 0
    max_time = 100  # Safety limit
    
    while not processor_manager.all_idle() and current_time < max_time:
        # Execute one time slice on all processors
        results = processor_manager.execute_all(current_time)
        
        # Update waiting times
        processor_manager.update_all_waiting_times()
        
        # Print progress every 5 time units
        if current_time % 5 == 0:
            active = sum(1 for r in results if r['executed'])
            completed = processor_manager.get_completed_count()
            print(f"  Time {current_time:3d}: {active} processors active, "
                  f"{completed}/{len(processes)} completed")
        
        current_time += 1
    
    # Calculate final metrics
    print(f"\n[Simulation Complete]")
    print("-" * 60)
    print(f"Total Time: {current_time} time units")
    print(f"Completed: {processor_manager.get_completed_count()}/{len(processes)}")
    
    # Process metrics
    completed_processes = [p for p in processes if p.is_completed()]
    
    if completed_processes:
        turnaround_times = [p.get_turnaround_time() for p in completed_processes]
        waiting_times = [p.waiting_time for p in completed_processes]
        
        print(f"\nProcess Metrics:")
        print(f"  Average Turnaround Time: {calculate_mean(turnaround_times):.2f}")
        print(f"  Average Waiting Time: {calculate_mean(waiting_times):.2f}")
    
    # Processor metrics
    utilizations = processor_manager.get_utilizations(current_time)
    
    print(f"\nProcessor Metrics:")
    for i, proc in enumerate(processor_manager):
        print(f"  Processor {i}: "
              f"Utilization={utilizations[i]*100:.1f}%, "
              f"Completed={proc.statistics.processes_completed}")
    
    print(f"\nOverall Metrics:")
    print(f"  Average Utilization: {calculate_mean(utilizations)*100:.1f}%")
    print(f"  Load Balance Index: {calculate_load_balance_index(processor_manager.get_loads()):.4f}")
    print(f"  Jain's Fairness Index: {calculate_jains_fairness_index(utilizations):.4f}")
    
    logger.info("CLI simulation completed")
    print("\n[CLI Simulation Complete]")


def run_module_tests():
    """Run tests for all modules."""
    print_banner()
    print("\n[Running Module Tests]\n")
    
    test_results = []
    
    # Test config module
    print("Testing config.py...")
    try:
        from config import SimulationConfig, GUIConfig
        config = SimulationConfig()
        config.validate()
        gui_config = GUIConfig()
        test_results.append(("config.py", "PASS"))
        print("  ✓ config.py: PASS")
    except Exception as e:
        test_results.append(("config.py", f"FAIL: {e}"))
        print(f"  ✗ config.py: FAIL - {e}")
    
    # Test process module
    print("Testing process.py...")
    try:
        from process import Process, ProcessGenerator, ProcessPriority
        p = Process(pid=1, arrival_time=0, burst_time=10)
        assert p.remaining_time == 10
        p.set_ready()
        assert p.is_ready()
        
        gen = ProcessGenerator()
        processes = gen.generate_processes(5)
        assert len(processes) == 5
        
        test_results.append(("process.py", "PASS"))
        print("  ✓ process.py: PASS")
    except Exception as e:
        test_results.append(("process.py", f"FAIL: {e}"))
        print(f"  ✗ process.py: FAIL - {e}")
    
    # Test processor module
    print("Testing processor.py...")
    try:
        from processor import Processor, ProcessorManager
        proc = Processor(processor_id=0)
        assert proc.is_idle()
        
        manager = ProcessorManager(num_processors=4)
        assert len(manager) == 4
        
        test_results.append(("processor.py", "PASS"))
        print("  ✓ processor.py: PASS")
    except Exception as e:
        test_results.append(("processor.py", f"FAIL: {e}"))
        print(f"  ✗ processor.py: FAIL - {e}")
    
    # Test utils module
    print("Testing utils.py...")
    try:
        from utils import (
            calculate_mean, 
            calculate_variance,
            calculate_load_balance_index,
            validate_positive_int
        )
        assert calculate_mean([1, 2, 3, 4, 5]) == 3.0
        assert validate_positive_int(5, "test") == 5
        
        test_results.append(("utils.py", "PASS"))
        print("  ✓ utils.py: PASS")
    except Exception as e:
        test_results.append(("utils.py", f"FAIL: {e}"))
        print(f"  ✗ utils.py: FAIL - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result == "PASS")
    total = len(test_results)
    
    for module, result in test_results:
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {module}: {result}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All module tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


def run_gui():
    """
    Run the graphical user interface.
    
    This imports and starts the GUI module.
    """
    print_banner()
    print("\n[Starting GUI Application]\n")
    
    try:
        # Import GUI module (will be created in Phase 4)
        from gui import LoadBalancerGUI
        
        # Create and run the application
        app = LoadBalancerGUI()
        app.run()
        
    except ImportError as e:
        print(f"Error: GUI module not found. {e}")
        print("\nThe GUI module (gui.py) will be implemented in Phase 4.")
        print("For now, you can run the CLI simulation with: python main.py --cli")
        print("Or run module tests with: python main.py --test")
        return 1
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    """
    Main entry point for the application.
    
    Parses command-line arguments and runs the appropriate mode.
    """
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Run the GUI application
  python main.py --cli        Run CLI simulation for testing
  python main.py --test       Run module tests
  python main.py --version    Show version information
        """
    )
    
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run in command-line mode (no GUI)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run module tests'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'{APP_NAME} v{VERSION}'
    )
    
    parser.add_argument(
        '--processors', '-p',
        type=int,
        default=4,
        help='Number of processors (default: 4)'
    )
    
    parser.add_argument(
        '--processes', '-n',
        type=int,
        default=20,
        help='Number of processes (default: 20)'
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        choices=['rr', 'round_robin', 'll', 'least_loaded', 'tb', 'threshold'],
        default='round_robin',
        help='Load balancing algorithm (default: round_robin)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run appropriate mode
    if args.test:
        return run_module_tests()
    elif args.cli:
        run_cli_simulation()
        return 0
    else:
        return run_gui()


if __name__ == "__main__":
    sys.exit(main())
