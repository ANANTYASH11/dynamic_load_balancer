"""
Metrics Module for Dynamic Load Balancing Simulator

This module provides comprehensive performance metric calculations for
analyzing the effectiveness of load balancing algorithms. Metrics are
essential for comparing different algorithms and evaluating system performance.

Key Metrics Implemented:
- Process Metrics: Turnaround time, waiting time, response time
- Processor Metrics: CPU utilization, throughput, load variance
- System Metrics: Fairness indices, migration statistics, efficiency

OS Concepts:
- Performance evaluation is crucial for OS design decisions
- These metrics help identify bottlenecks and optimization opportunities
- Trade-offs between different metrics guide algorithm selection

Author: Student
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import statistics
import math
import json
from datetime import datetime

from config import (
    ProcessState,
    LoadBalancingAlgorithm,
    SimulationConfig,
    DEFAULT_SIMULATION_CONFIG
)


@dataclass
class ProcessMetrics:
    """
    Metrics for a single process.
    
    These metrics capture the process's experience in the system:
    - How long it waited before starting
    - How long it took to complete
    - How responsive the system was
    """
    pid: int
    arrival_time: int
    burst_time: int
    start_time: Optional[int]
    completion_time: Optional[int]
    waiting_time: int
    processor_id: Optional[int]
    migration_count: int
    priority: str
    
    @property
    def turnaround_time(self) -> Optional[int]:
        """
        Turnaround Time = Completion Time - Arrival Time
        
        Total time the process spent in the system, from arrival to completion.
        Includes both waiting time and execution time.
        """
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None
    
    @property
    def response_time(self) -> Optional[int]:
        """
        Response Time = Start Time - Arrival Time
        
        Time from process arrival until it first gets CPU time.
        Important for interactive systems where quick response matters.
        """
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None
    
    @property
    def normalized_turnaround(self) -> Optional[float]:
        """
        Normalized Turnaround Time = Turnaround Time / Burst Time
        
        Ratio that accounts for process size. A value of 1.0 means the
        process completed without any waiting (ideal case).
        """
        if self.turnaround_time is not None and self.burst_time > 0:
            return self.turnaround_time / self.burst_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'pid': self.pid,
            'arrival_time': self.arrival_time,
            'burst_time': self.burst_time,
            'start_time': self.start_time,
            'completion_time': self.completion_time,
            'waiting_time': self.waiting_time,
            'turnaround_time': self.turnaround_time,
            'response_time': self.response_time,
            'normalized_turnaround': self.normalized_turnaround,
            'processor_id': self.processor_id,
            'migration_count': self.migration_count,
            'priority': self.priority
        }


@dataclass
class ProcessorMetrics:
    """
    Metrics for a single processor.
    
    These metrics evaluate how effectively the processor was utilized:
    - High utilization indicates good workload assignment
    - Low idle time shows efficient scheduling
    - Context switches indicate scheduling overhead
    """
    processor_id: int
    total_execution_time: int
    total_idle_time: int
    total_context_switches: int
    processes_completed: int
    processes_received: int
    processes_migrated_in: int
    processes_migrated_out: int
    final_queue_size: int
    
    def get_utilization(self, total_time: int) -> float:
        """
        CPU Utilization = Execution Time / Total Time
        
        Percentage of time the processor was actively executing.
        Higher is generally better (less wasted resources).
        """
        if total_time <= 0:
            return 0.0
        return min(1.0, self.total_execution_time / total_time)
    
    def get_throughput(self, total_time: int) -> float:
        """
        Throughput = Processes Completed / Total Time
        
        Number of processes completed per time unit.
        Higher throughput indicates better productivity.
        """
        if total_time <= 0:
            return 0.0
        return self.processes_completed / total_time
    
    def get_context_switch_rate(self, total_time: int) -> float:
        """
        Context Switch Rate = Context Switches / Total Time
        
        Frequency of process switches. Too high indicates excessive overhead.
        """
        if total_time <= 0:
            return 0.0
        return self.total_context_switches / total_time
    
    def to_dict(self, total_time: int = 0) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'processor_id': self.processor_id,
            'total_execution_time': self.total_execution_time,
            'total_idle_time': self.total_idle_time,
            'total_context_switches': self.total_context_switches,
            'processes_completed': self.processes_completed,
            'processes_received': self.processes_received,
            'processes_migrated_in': self.processes_migrated_in,
            'processes_migrated_out': self.processes_migrated_out,
            'final_queue_size': self.final_queue_size,
            'utilization': self.get_utilization(total_time) if total_time > 0 else 0,
            'throughput': self.get_throughput(total_time) if total_time > 0 else 0
        }


@dataclass
class SystemMetrics:
    """
    Aggregate metrics for the entire system.
    
    These metrics evaluate the overall effectiveness of load balancing:
    - Fairness measures how evenly work was distributed
    - Load balance metrics show distribution quality
    - Migration stats indicate algorithm activity
    """
    # Time metrics
    total_simulation_time: int = 0
    total_processes: int = 0
    completed_processes: int = 0
    
    # Process aggregate metrics
    avg_turnaround_time: float = 0.0
    avg_waiting_time: float = 0.0
    avg_response_time: float = 0.0
    avg_normalized_turnaround: float = 0.0
    
    max_turnaround_time: int = 0
    max_waiting_time: int = 0
    min_turnaround_time: int = 0
    min_waiting_time: int = 0
    
    # Processor aggregate metrics
    num_processors: int = 0
    avg_utilization: float = 0.0
    min_utilization: float = 0.0
    max_utilization: float = 0.0
    utilization_std_dev: float = 0.0
    
    total_throughput: float = 0.0
    total_context_switches: int = 0
    
    # Load balance metrics
    load_variance: float = 0.0
    load_std_dev: float = 0.0
    load_balance_index: float = 0.0
    jains_fairness_index: float = 0.0
    coefficient_of_variation: float = 0.0
    
    # Migration metrics
    total_migrations: int = 0
    migration_rate: float = 0.0
    processes_migrated: int = 0
    
    # Algorithm info
    algorithm_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'simulation_time': self.total_simulation_time,
            'total_processes': self.total_processes,
            'completed_processes': self.completed_processes,
            'completion_rate': self.completed_processes / self.total_processes if self.total_processes > 0 else 0,
            
            'avg_turnaround_time': round(self.avg_turnaround_time, 2),
            'avg_waiting_time': round(self.avg_waiting_time, 2),
            'avg_response_time': round(self.avg_response_time, 2),
            'avg_normalized_turnaround': round(self.avg_normalized_turnaround, 2),
            
            'max_turnaround_time': self.max_turnaround_time,
            'max_waiting_time': self.max_waiting_time,
            'min_turnaround_time': self.min_turnaround_time,
            'min_waiting_time': self.min_waiting_time,
            
            'num_processors': self.num_processors,
            'avg_utilization': round(self.avg_utilization * 100, 2),
            'min_utilization': round(self.min_utilization * 100, 2),
            'max_utilization': round(self.max_utilization * 100, 2),
            'utilization_std_dev': round(self.utilization_std_dev * 100, 2),
            
            'total_throughput': round(self.total_throughput, 4),
            'total_context_switches': self.total_context_switches,
            
            'load_variance': round(self.load_variance, 4),
            'load_std_dev': round(self.load_std_dev, 4),
            'load_balance_index': round(self.load_balance_index, 4),
            'jains_fairness_index': round(self.jains_fairness_index, 4),
            'coefficient_of_variation': round(self.coefficient_of_variation, 4),
            
            'total_migrations': self.total_migrations,
            'migration_rate': round(self.migration_rate, 4),
            'processes_migrated': self.processes_migrated,
            
            'algorithm': self.algorithm_name
        }


class MetricsCalculator:
    """
    Calculator class for computing all performance metrics.
    
    This class centralizes metric calculations and provides:
    - Per-process metrics
    - Per-processor metrics  
    - System-wide aggregate metrics
    - Comparison functionality for different algorithms
    
    Usage:
        calculator = MetricsCalculator()
        calculator.collect_process_metrics(processes)
        calculator.collect_processor_metrics(processors, total_time)
        system_metrics = calculator.calculate_system_metrics()
    """
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = None):
        """
        Initialize the metrics calculator.
        
        Args:
            algorithm: The load balancing algorithm being evaluated
        """
        self.algorithm = algorithm
        self.process_metrics: List[ProcessMetrics] = []
        self.processor_metrics: List[ProcessorMetrics] = []
        self.system_metrics: Optional[SystemMetrics] = None
        self.total_time: int = 0
        
        # Time series data for charts
        self.utilization_history: List[List[float]] = []  # [time][processor]
        self.load_history: List[List[float]] = []  # [time][processor]
        self.queue_history: List[List[int]] = []  # [time][processor]
        self.completed_history: List[int] = []  # [time] -> cumulative completed
    
    def reset(self):
        """Reset all collected metrics."""
        self.process_metrics.clear()
        self.processor_metrics.clear()
        self.system_metrics = None
        self.total_time = 0
        self.utilization_history.clear()
        self.load_history.clear()
        self.queue_history.clear()
        self.completed_history.clear()
    
    def collect_process_metrics(self, processes: List[Any]) -> List[ProcessMetrics]:
        """
        Collect metrics from a list of Process objects.
        
        Args:
            processes: List of Process objects
            
        Returns:
            List of ProcessMetrics objects
        """
        self.process_metrics.clear()
        
        for p in processes:
            metrics = ProcessMetrics(
                pid=p.pid,
                arrival_time=p.arrival_time,
                burst_time=p.burst_time,
                start_time=p.start_time,
                completion_time=p.completion_time,
                waiting_time=p.waiting_time,
                processor_id=p.processor_id,
                migration_count=p.migration_count,
                priority=p.priority.name if hasattr(p.priority, 'name') else str(p.priority)
            )
            self.process_metrics.append(metrics)
        
        return self.process_metrics
    
    def collect_processor_metrics(self, processors: List[Any], total_time: int) -> List[ProcessorMetrics]:
        """
        Collect metrics from a list of Processor objects.
        
        Args:
            processors: List of Processor objects
            total_time: Total simulation time
            
        Returns:
            List of ProcessorMetrics objects
        """
        self.processor_metrics.clear()
        self.total_time = total_time
        
        for proc in processors:
            stats = proc.statistics
            metrics = ProcessorMetrics(
                processor_id=proc.processor_id,
                total_execution_time=stats.total_execution_time,
                total_idle_time=stats.total_idle_time,
                total_context_switches=stats.total_context_switches,
                processes_completed=stats.processes_completed,
                processes_received=stats.processes_received,
                processes_migrated_in=stats.processes_migrated_in,
                processes_migrated_out=stats.processes_migrated_out,
                final_queue_size=proc.get_queue_size()
            )
            self.processor_metrics.append(metrics)
        
        return self.processor_metrics
    
    def record_time_point(self, processors: List[Any], completed_count: int):
        """
        Record metrics at a specific time point for time series analysis.
        
        Args:
            processors: List of Processor objects
            completed_count: Number of completed processes
        """
        utilizations = []
        loads = []
        queues = []
        
        for proc in processors:
            # Calculate instantaneous utilization (simplified)
            util = 1.0 if proc.current_process else 0.0
            utilizations.append(util)
            loads.append(proc.get_load())
            queues.append(proc.get_queue_size())
        
        self.utilization_history.append(utilizations)
        self.load_history.append(loads)
        self.queue_history.append(queues)
        self.completed_history.append(completed_count)
    
    def calculate_system_metrics(self) -> SystemMetrics:
        """
        Calculate aggregate system metrics from collected data.
        
        Returns:
            SystemMetrics object with all calculated values
        """
        metrics = SystemMetrics()
        metrics.algorithm_name = self.algorithm.value if self.algorithm else "Unknown"
        metrics.total_simulation_time = self.total_time
        metrics.total_processes = len(self.process_metrics)
        metrics.num_processors = len(self.processor_metrics)
        
        # Calculate process metrics
        completed = [p for p in self.process_metrics if p.completion_time is not None]
        metrics.completed_processes = len(completed)
        
        if completed:
            turnaround_times = [p.turnaround_time for p in completed if p.turnaround_time is not None]
            waiting_times = [p.waiting_time for p in completed]
            response_times = [p.response_time for p in completed if p.response_time is not None]
            normalized = [p.normalized_turnaround for p in completed if p.normalized_turnaround is not None]
            
            if turnaround_times:
                metrics.avg_turnaround_time = statistics.mean(turnaround_times)
                metrics.max_turnaround_time = max(turnaround_times)
                metrics.min_turnaround_time = min(turnaround_times)
            
            if waiting_times:
                metrics.avg_waiting_time = statistics.mean(waiting_times)
                metrics.max_waiting_time = max(waiting_times)
                metrics.min_waiting_time = min(waiting_times)
            
            if response_times:
                metrics.avg_response_time = statistics.mean(response_times)
            
            if normalized:
                metrics.avg_normalized_turnaround = statistics.mean(normalized)
        
        # Calculate processor metrics
        if self.processor_metrics and self.total_time > 0:
            utilizations = [p.get_utilization(self.total_time) for p in self.processor_metrics]
            
            metrics.avg_utilization = statistics.mean(utilizations)
            metrics.min_utilization = min(utilizations)
            metrics.max_utilization = max(utilizations)
            
            if len(utilizations) > 1:
                metrics.utilization_std_dev = statistics.stdev(utilizations)
            
            metrics.total_throughput = sum(p.get_throughput(self.total_time) for p in self.processor_metrics)
            metrics.total_context_switches = sum(p.total_context_switches for p in self.processor_metrics)
            
            # Load balance metrics
            metrics.load_variance = self._calculate_variance(utilizations)
            metrics.load_std_dev = math.sqrt(metrics.load_variance)
            metrics.load_balance_index = self._calculate_load_balance_index(utilizations)
            metrics.jains_fairness_index = self._calculate_jains_fairness(utilizations)
            metrics.coefficient_of_variation = self._calculate_cv(utilizations)
            
            # Migration metrics
            metrics.total_migrations = sum(p.processes_migrated_in for p in self.processor_metrics)
            metrics.processes_migrated = len([p for p in self.process_metrics if p.migration_count > 0])
            if self.total_time > 0:
                metrics.migration_rate = metrics.total_migrations / self.total_time
        
        self.system_metrics = metrics
        return metrics
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate population variance."""
        if len(values) < 2:
            return 0.0
        mean = statistics.mean(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _calculate_load_balance_index(self, loads: List[float]) -> float:
        """
        Calculate Load Balance Index.
        
        LBI = 1 - (max - min) / max
        Range: 0 (worst) to 1 (perfect balance)
        """
        if not loads or max(loads) == 0:
            return 1.0
        max_load = max(loads)
        min_load = min(loads)
        return 1 - (max_load - min_load) / max_load if max_load > 0 else 1.0
    
    def _calculate_jains_fairness(self, values: List[float]) -> float:
        """
        Calculate Jain's Fairness Index.
        
        J(x) = (Σxi)² / (n × Σxi²)
        Range: 1/n (worst) to 1 (perfect fairness)
        """
        if not values:
            return 1.0
        n = len(values)
        sum_values = sum(values)
        sum_squares = sum(x ** 2 for x in values)
        if sum_squares == 0:
            return 1.0
        return (sum_values ** 2) / (n * sum_squares)
    
    def _calculate_cv(self, values: List[float]) -> float:
        """
        Calculate Coefficient of Variation.
        
        CV = σ / μ
        Lower is better (less variation relative to mean)
        """
        if not values:
            return 0.0
        mean = statistics.mean(values)
        if mean == 0:
            return 0.0
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        return std_dev / mean
    
    def get_process_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of process metrics."""
        completed = [p for p in self.process_metrics if p.completion_time is not None]
        
        if not completed:
            return {'count': 0}
        
        turnaround = [p.turnaround_time for p in completed if p.turnaround_time]
        waiting = [p.waiting_time for p in completed]
        
        return {
            'count': len(completed),
            'turnaround': {
                'avg': statistics.mean(turnaround) if turnaround else 0,
                'min': min(turnaround) if turnaround else 0,
                'max': max(turnaround) if turnaround else 0,
                'std': statistics.stdev(turnaround) if len(turnaround) > 1 else 0
            },
            'waiting': {
                'avg': statistics.mean(waiting) if waiting else 0,
                'min': min(waiting) if waiting else 0,
                'max': max(waiting) if waiting else 0,
                'std': statistics.stdev(waiting) if len(waiting) > 1 else 0
            }
        }
    
    def get_processor_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of processor metrics."""
        if not self.processor_metrics or self.total_time == 0:
            return {'count': 0}
        
        utilizations = [p.get_utilization(self.total_time) for p in self.processor_metrics]
        
        return {
            'count': len(self.processor_metrics),
            'utilization': {
                'avg': statistics.mean(utilizations),
                'min': min(utilizations),
                'max': max(utilizations),
                'std': statistics.stdev(utilizations) if len(utilizations) > 1 else 0
            },
            'total_completed': sum(p.processes_completed for p in self.processor_metrics),
            'total_context_switches': sum(p.total_context_switches for p in self.processor_metrics)
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all metrics to a dictionary."""
        return {
            'timestamp': datetime.now().isoformat(),
            'algorithm': self.algorithm.value if self.algorithm else 'Unknown',
            'system': self.system_metrics.to_dict() if self.system_metrics else {},
            'processes': [p.to_dict() for p in self.process_metrics],
            'processors': [p.to_dict(self.total_time) for p in self.processor_metrics]
        }
    
    def export_to_json(self, filepath: str) -> str:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to output file
            
        Returns:
            Path to created file
        """
        data = self.export_to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return filepath


class MetricsComparator:
    """
    Compare metrics across different algorithm runs.
    
    This class helps evaluate which algorithm performs best
    for a given workload by comparing key metrics.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.results: Dict[str, SystemMetrics] = {}
    
    def add_result(self, algorithm: LoadBalancingAlgorithm, metrics: SystemMetrics):
        """
        Add results for an algorithm.
        
        Args:
            algorithm: The algorithm that was tested
            metrics: The resulting system metrics
        """
        self.results[algorithm.value] = metrics
    
    def clear(self):
        """Clear all stored results."""
        self.results.clear()
    
    def get_comparison_table(self) -> List[Dict[str, Any]]:
        """
        Generate a comparison table of all algorithms.
        
        Returns:
            List of dictionaries with comparison data
        """
        comparison = []
        
        for algo_name, metrics in self.results.items():
            comparison.append({
                'Algorithm': algo_name,
                'Avg Turnaround': round(metrics.avg_turnaround_time, 2),
                'Avg Waiting': round(metrics.avg_waiting_time, 2),
                'Avg Utilization': f"{metrics.avg_utilization*100:.1f}%",
                'Load Balance Index': round(metrics.load_balance_index, 4),
                'Fairness Index': round(metrics.jains_fairness_index, 4),
                'Migrations': metrics.total_migrations,
                'Context Switches': metrics.total_context_switches
            })
        
        return comparison
    
    def get_best_algorithm(self, metric: str = 'avg_turnaround_time') -> Optional[str]:
        """
        Determine the best algorithm based on a specific metric.
        
        Args:
            metric: Name of the metric to compare
            
        Returns:
            Name of the best performing algorithm
        """
        if not self.results:
            return None
        
        # Metrics where lower is better
        lower_is_better = {
            'avg_turnaround_time', 'avg_waiting_time', 'avg_response_time',
            'load_variance', 'load_std_dev', 'coefficient_of_variation',
            'total_migrations'
        }
        
        # Metrics where higher is better
        higher_is_better = {
            'avg_utilization', 'load_balance_index', 'jains_fairness_index',
            'total_throughput', 'completed_processes'
        }
        
        best_algo = None
        best_value = None
        
        for algo_name, metrics in self.results.items():
            value = getattr(metrics, metric, None)
            if value is None:
                continue
            
            if best_value is None:
                best_algo = algo_name
                best_value = value
            elif metric in lower_is_better and value < best_value:
                best_algo = algo_name
                best_value = value
            elif metric in higher_is_better and value > best_value:
                best_algo = algo_name
                best_value = value
        
        return best_algo
    
    def get_ranking(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get ranking of algorithms for each metric.
        
        Returns:
            Dictionary mapping metric names to ranked algorithm lists
        """
        if not self.results:
            return {}
        
        metrics_to_rank = [
            'avg_turnaround_time',
            'avg_waiting_time',
            'avg_utilization',
            'load_balance_index',
            'jains_fairness_index',
            'total_migrations'
        ]
        
        rankings = {}
        
        for metric in metrics_to_rank:
            values = []
            for algo_name, sys_metrics in self.results.items():
                value = getattr(sys_metrics, metric, 0)
                values.append((algo_name, value))
            
            # Sort based on whether lower or higher is better
            lower_is_better = metric in {
                'avg_turnaround_time', 'avg_waiting_time', 'total_migrations'
            }
            values.sort(key=lambda x: x[1], reverse=not lower_is_better)
            rankings[metric] = values
        
        return rankings
    
    def generate_report(self) -> str:
        """
        Generate a text report comparing all algorithms.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No results to compare."
        
        lines = []
        lines.append("=" * 70)
        lines.append("ALGORITHM COMPARISON REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Comparison table
        table = self.get_comparison_table()
        if table:
            headers = list(table[0].keys())
            
            # Calculate column widths
            widths = {h: len(h) for h in headers}
            for row in table:
                for h in headers:
                    widths[h] = max(widths[h], len(str(row[h])))
            
            # Print header
            header_line = " | ".join(h.ljust(widths[h]) for h in headers)
            lines.append(header_line)
            lines.append("-" * len(header_line))
            
            # Print rows
            for row in table:
                row_line = " | ".join(str(row[h]).ljust(widths[h]) for h in headers)
                lines.append(row_line)
        
        lines.append("")
        lines.append("-" * 70)
        lines.append("RANKINGS")
        lines.append("-" * 70)
        
        rankings = self.get_ranking()
        for metric, ranked in rankings.items():
            metric_name = metric.replace('_', ' ').title()
            lines.append(f"\n{metric_name}:")
            for i, (algo, value) in enumerate(ranked, 1):
                lines.append(f"  {i}. {algo}: {value:.4f}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Metrics Module Test")
    print("=" * 60)
    
    # Create test data
    from process import ProcessGenerator
    from processor import ProcessorManager
    
    print("\n1. Testing MetricsCalculator:")
    
    # Generate test processes
    generator = ProcessGenerator()
    processes = generator.generate_predefined_test_set()
    
    # Simulate completion for some processes
    for i, p in enumerate(processes):
        p.processor_id = i % 4
        if i < 8:  # Complete 8 of 10 processes
            p.start_time = p.arrival_time + 2
            p.completion_time = p.start_time + p.burst_time
            p.waiting_time = p.start_time - p.arrival_time
            p.state = ProcessState.COMPLETED
    
    # Create processor manager and simulate
    manager = ProcessorManager(num_processors=4)
    for proc in manager:
        proc.statistics.total_execution_time = 20
        proc.statistics.total_idle_time = 5
        proc.statistics.processes_completed = 2
        proc.statistics.processes_received = 3
        proc.statistics.total_context_switches = 4
    
    # Calculate metrics
    calculator = MetricsCalculator(LoadBalancingAlgorithm.ROUND_ROBIN)
    calculator.collect_process_metrics(processes)
    calculator.collect_processor_metrics(list(manager), 25)
    system_metrics = calculator.calculate_system_metrics()
    
    print(f"\n   System Metrics:")
    print(f"   - Total Time: {system_metrics.total_simulation_time}")
    print(f"   - Completed: {system_metrics.completed_processes}/{system_metrics.total_processes}")
    print(f"   - Avg Turnaround: {system_metrics.avg_turnaround_time:.2f}")
    print(f"   - Avg Waiting: {system_metrics.avg_waiting_time:.2f}")
    print(f"   - Avg Utilization: {system_metrics.avg_utilization*100:.1f}%")
    print(f"   - Load Balance Index: {system_metrics.load_balance_index:.4f}")
    print(f"   - Jain's Fairness: {system_metrics.jains_fairness_index:.4f}")
    
    print("\n2. Testing MetricsComparator:")
    comparator = MetricsComparator()
    
    # Add fake results for different algorithms
    comparator.add_result(LoadBalancingAlgorithm.ROUND_ROBIN, system_metrics)
    
    # Create slightly different metrics for other algorithms
    system_metrics_ll = SystemMetrics(
        avg_turnaround_time=10.5,
        avg_waiting_time=4.2,
        avg_utilization=0.82,
        load_balance_index=0.92,
        jains_fairness_index=0.95,
        total_migrations=0
    )
    comparator.add_result(LoadBalancingAlgorithm.LEAST_LOADED, system_metrics_ll)
    
    system_metrics_tb = SystemMetrics(
        avg_turnaround_time=11.0,
        avg_waiting_time=4.5,
        avg_utilization=0.78,
        load_balance_index=0.95,
        jains_fairness_index=0.97,
        total_migrations=5
    )
    comparator.add_result(LoadBalancingAlgorithm.THRESHOLD_BASED, system_metrics_tb)
    
    print("\n   Comparison Table:")
    for row in comparator.get_comparison_table():
        print(f"   {row}")
    
    print(f"\n   Best for turnaround: {comparator.get_best_algorithm('avg_turnaround_time')}")
    print(f"   Best for fairness: {comparator.get_best_algorithm('jains_fairness_index')}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
