"""
Utility Module for Dynamic Load Balancing Simulator

This module contains helper functions and utility classes used throughout
the simulation. Centralizing utilities promotes code reuse and maintains
consistency across modules.

Contents:
- Logging configuration and utilities
- Time formatting functions
- Statistical calculations
- Data export functions
- Validation helpers

Author: Student
Date: December 2024
"""

import logging
import sys
import os
import json
import csv
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
from pathlib import Path

from config import (
    LoggingConfig,
    DEFAULT_LOGGING_CONFIG,
    ProcessState,
    LoadBalancingAlgorithm
)


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class SimulationLogger:
    """
    Custom logger for the simulation with support for both console and file output.
    
    Features:
    - Colored console output for different log levels
    - Structured log file output
    - Configurable verbosity
    - Event-specific logging methods
    """
    
    # ANSI color codes for console output
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, name: str = "LoadBalancer", config: LoggingConfig = None):
        """
        Initialize the simulation logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.config = config or DEFAULT_LOGGING_CONFIG
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        console_formatter = ColoredFormatter(self.config.log_format)
        file_formatter = logging.Formatter(
            self.config.log_format,
            datefmt=self.config.date_format
        )
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            file_handler = logging.FileHandler(
                self.config.log_file_path,
                mode='w',  # Overwrite for each run
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_simulation_start(self, config_dict: Dict[str, Any]) -> None:
        """Log simulation start with configuration."""
        self.logger.info("=" * 60)
        self.logger.info("SIMULATION STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration: {json.dumps(config_dict, indent=2, default=str)}")
    
    def log_simulation_end(self, total_time: int, metrics: Dict[str, Any]) -> None:
        """Log simulation end with final metrics."""
        self.logger.info("=" * 60)
        self.logger.info(f"SIMULATION COMPLETED at time {total_time}")
        self.logger.info("=" * 60)
        self.logger.info(f"Final Metrics: {json.dumps(metrics, indent=2, default=str)}")
    
    def log_process_created(self, pid: int, burst: int, arrival: int, priority: str) -> None:
        """Log process creation."""
        if self.config.log_process_creation:
            self.logger.debug(
                f"Process Created: P{pid} (burst={burst}, arrival={arrival}, priority={priority})"
            )
    
    def log_process_assigned(self, pid: int, processor_id: int, algorithm: str) -> None:
        """Log process assignment to processor."""
        self.logger.info(
            f"Process Assigned: P{pid} -> Processor {processor_id} [{algorithm}]"
        )
    
    def log_process_completed(self, pid: int, processor_id: int, 
                              turnaround: int, waiting: int) -> None:
        """Log process completion."""
        if self.config.log_process_completion:
            self.logger.info(
                f"Process Completed: P{pid} on Processor {processor_id} "
                f"(turnaround={turnaround}, waiting={waiting})"
            )
    
    def log_process_migration(self, pid: int, from_proc: int, to_proc: int, reason: str) -> None:
        """Log process migration between processors."""
        if self.config.log_process_migration:
            self.logger.warning(
                f"Process Migration: P{pid} from Processor {from_proc} -> {to_proc} "
                f"(reason: {reason})"
            )
    
    def log_load_balance_decision(self, algorithm: str, decision: str, 
                                   loads: List[float]) -> None:
        """Log load balancing decision."""
        if self.config.log_load_balancing_decisions:
            load_str = ", ".join([f"P{i}:{l:.2f}" for i, l in enumerate(loads)])
            self.logger.info(
                f"Load Balance [{algorithm}]: {decision} | Loads: [{load_str}]"
            )
    
    def log_processor_state(self, processor_id: int, queue_size: int, 
                            current_pid: Optional[int], utilization: float) -> None:
        """Log processor state change."""
        if self.config.log_processor_state_changes:
            current = f"P{current_pid}" if current_pid else "Idle"
            self.logger.debug(
                f"Processor {processor_id}: queue={queue_size}, "
                f"current={current}, util={utilization*100:.1f}%"
            )
    
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)
    
    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(name: str = "LoadBalancer", config: LoggingConfig = None) -> SimulationLogger:
    """
    Set up and return the simulation logger.
    
    Args:
        name: Logger name
        config: Logging configuration
        
    Returns:
        Configured SimulationLogger instance
    """
    return SimulationLogger(name=name, config=config)


# =============================================================================
# TIME UTILITIES
# =============================================================================

def format_time(time_units: int, unit_duration_ms: int = 100) -> str:
    """
    Format simulation time units as human-readable string.
    
    Args:
        time_units: Number of simulation time units
        unit_duration_ms: Duration of each time unit in milliseconds
        
    Returns:
        Formatted time string
    """
    total_ms = time_units * unit_duration_ms
    
    if total_ms < 1000:
        return f"{total_ms}ms"
    elif total_ms < 60000:
        return f"{total_ms/1000:.1f}s"
    else:
        minutes = total_ms // 60000
        seconds = (total_ms % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def get_timestamp() -> str:
    """
    Get current timestamp for logging.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds as human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def calculate_mean(values: List[float]) -> float:
    """
    Calculate arithmetic mean.
    
    Args:
        values: List of numeric values
        
    Returns:
        Mean value (0 if list is empty)
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_variance(values: List[float]) -> float:
    """
    Calculate population variance.
    
    Args:
        values: List of numeric values
        
    Returns:
        Variance value (0 if list has fewer than 2 elements)
    """
    if len(values) < 2:
        return 0.0
    mean = calculate_mean(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def calculate_std_dev(values: List[float]) -> float:
    """
    Calculate population standard deviation.
    
    Args:
        values: List of numeric values
        
    Returns:
        Standard deviation
    """
    return calculate_variance(values) ** 0.5


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate percentile value.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Percentile value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100
    lower = int(index)
    upper = lower + 1
    
    if upper >= len(sorted_values):
        return sorted_values[-1]
    
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def calculate_load_balance_index(loads: List[float]) -> float:
    """
    Calculate Load Balance Index (LBI).
    
    LBI = 1 - (max_load - min_load) / max_load
    
    Perfect balance = 1.0, completely unbalanced = 0.0
    
    Args:
        loads: List of load values for each processor
        
    Returns:
        Load balance index (0.0 to 1.0)
    """
    if not loads or max(loads) == 0:
        return 1.0  # No load = perfectly balanced
    
    max_load = max(loads)
    min_load = min(loads)
    
    if max_load == 0:
        return 1.0
    
    return 1 - (max_load - min_load) / max_load


def calculate_jains_fairness_index(values: List[float]) -> float:
    """
    Calculate Jain's Fairness Index.
    
    A widely used metric for measuring fairness in resource allocation.
    J(x) = (Σxi)² / (n × Σxi²)
    
    Range: 1/n (worst) to 1 (perfect fairness)
    
    Args:
        values: List of values (e.g., utilizations, loads)
        
    Returns:
        Jain's fairness index (1/n to 1.0)
    """
    if not values:
        return 1.0
    
    n = len(values)
    sum_values = sum(values)
    sum_squares = sum(x ** 2 for x in values)
    
    if sum_squares == 0:
        return 1.0
    
    return (sum_values ** 2) / (n * sum_squares)


# =============================================================================
# DATA EXPORT UTILITIES
# =============================================================================

class DataExporter:
    """
    Export simulation data to various formats for analysis.
    
    Supports:
    - JSON export for raw data
    - CSV export for spreadsheet analysis
    - Summary text reports
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the data exporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data to JSON file.
        
        Args:
            data: Data dictionary to export
            filename: Output filename (without extension)
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / f"{filename}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        return str(filepath)
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: str) -> str:
        """
        Export list of records to CSV file.
        
        Args:
            data: List of dictionaries to export
            filename: Output filename (without extension)
            
        Returns:
            Path to created file
        """
        if not data:
            return ""
        
        filepath = self.output_dir / f"{filename}.csv"
        
        # Get all unique keys
        fieldnames = set()
        for record in data:
            fieldnames.update(record.keys())
        fieldnames = sorted(fieldnames)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        return str(filepath)
    
    def export_process_data(self, processes: List[Any], filename: str = "processes") -> str:
        """
        Export process data to CSV.
        
        Args:
            processes: List of Process objects
            filename: Output filename
            
        Returns:
            Path to created file
        """
        data = []
        for p in processes:
            data.append({
                'PID': p.pid,
                'Arrival Time': p.arrival_time,
                'Burst Time': p.burst_time,
                'Priority': p.priority.name if hasattr(p.priority, 'name') else p.priority,
                'Processor ID': p.processor_id,
                'Start Time': p.start_time,
                'Completion Time': p.completion_time,
                'Waiting Time': p.waiting_time,
                'Turnaround Time': p.get_turnaround_time(),
                'Response Time': p.get_response_time(),
                'Migration Count': p.migration_count
            })
        
        return self.export_to_csv(data, filename)
    
    def export_summary_report(self, metrics: Dict[str, Any], 
                               config: Dict[str, Any],
                               filename: str = "summary") -> str:
        """
        Export summary report as text file.
        
        Args:
            metrics: Performance metrics dictionary
            config: Configuration dictionary
            filename: Output filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / f"{filename}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("DYNAMIC LOAD BALANCING SIMULATION - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 70 + "\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                elif isinstance(value, dict):
                    f.write(f"  {key}:\n")
                    for k, v in value.items():
                        f.write(f"    {k}: {v}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")
        
        return str(filepath)


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_positive_int(value: Any, name: str, min_val: int = 1, 
                          max_val: int = None) -> int:
    """
    Validate that a value is a positive integer within range.
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        min_val: Minimum allowed value
        max_val: Maximum allowed value (None for no limit)
        
    Returns:
        Validated integer value
        
    Raises:
        ValueError: If validation fails
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    
    if int_value < min_val:
        raise ValueError(f"{name} must be at least {min_val}, got {int_value}")
    
    if max_val is not None and int_value > max_val:
        raise ValueError(f"{name} must be at most {max_val}, got {int_value}")
    
    return int_value


def validate_positive_float(value: Any, name: str, min_val: float = 0.0,
                            max_val: float = None) -> float:
    """
    Validate that a value is a positive float within range.
    
    Args:
        value: Value to validate
        name: Name of the parameter
        min_val: Minimum allowed value
        max_val: Maximum allowed value (None for no limit)
        
    Returns:
        Validated float value
        
    Raises:
        ValueError: If validation fails
    """
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    
    if float_value < min_val:
        raise ValueError(f"{name} must be at least {min_val}, got {float_value}")
    
    if max_val is not None and float_value > max_val:
        raise ValueError(f"{name} must be at most {max_val}, got {float_value}")
    
    return float_value


def validate_algorithm(value: str) -> LoadBalancingAlgorithm:
    """
    Validate and convert algorithm string to enum.
    
    Args:
        value: Algorithm name or value
        
    Returns:
        LoadBalancingAlgorithm enum value
        
    Raises:
        ValueError: If algorithm is not recognized
    """
    # Check if it's already an enum
    if isinstance(value, LoadBalancingAlgorithm):
        return value
    
    # Try to match by value (display name)
    for algo in LoadBalancingAlgorithm:
        if algo.value.lower() == value.lower():
            return algo
    
    # Try to match by name
    for algo in LoadBalancingAlgorithm:
        if algo.name.lower() == value.lower().replace(" ", "_"):
            return algo
    
    valid_options = [algo.value for algo in LoadBalancingAlgorithm]
    raise ValueError(f"Invalid algorithm '{value}'. Valid options: {valid_options}")


# =============================================================================
# MISCELLANEOUS UTILITIES
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a specified range.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def interpolate_color(color1: str, color2: str, factor: float) -> str:
    """
    Interpolate between two hex colors.
    
    Args:
        color1: Start color (hex string like "#RRGGBB")
        color2: End color (hex string)
        factor: Interpolation factor (0.0 = color1, 1.0 = color2)
        
    Returns:
        Interpolated color as hex string
    """
    # Parse hex colors
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    
    # Interpolate
    factor = clamp(factor, 0.0, 1.0)
    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)
    
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_process_color(pid: int) -> str:
    """
    Generate a unique color for a process based on its PID.
    
    Args:
        pid: Process ID
        
    Returns:
        Hex color string
    """
    # Use golden ratio for good color distribution
    golden_ratio = 0.618033988749895
    hue = (pid * golden_ratio) % 1.0
    
    # Convert HSV to RGB (saturation=0.7, value=0.9)
    h = hue * 6
    c = 0.9 * 0.7
    x = c * (1 - abs(h % 2 - 1))
    m = 0.9 - c
    
    if h < 1:
        r, g, b = c, x, 0
    elif h < 2:
        r, g, b = x, c, 0
    elif h < 3:
        r, g, b = 0, c, x
    elif h < 4:
        r, g, b = 0, x, c
    elif h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Utility Module Test")
    print("=" * 60)
    
    # Test logger
    print("\n1. Testing SimulationLogger:")
    logger = SimulationLogger(name="TestLogger")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.debug("This is a debug message")
    
    # Test statistical functions
    print("\n2. Testing statistical functions:")
    test_values = [10, 20, 30, 40, 50]
    print(f"   Values: {test_values}")
    print(f"   Mean: {calculate_mean(test_values)}")
    print(f"   Variance: {calculate_variance(test_values)}")
    print(f"   Std Dev: {calculate_std_dev(test_values):.2f}")
    print(f"   50th percentile: {calculate_percentile(test_values, 50)}")
    
    # Test load balance metrics
    print("\n3. Testing load balance metrics:")
    loads = [0.8, 0.75, 0.7, 0.65]
    print(f"   Loads: {loads}")
    print(f"   Load Balance Index: {calculate_load_balance_index(loads):.4f}")
    print(f"   Jain's Fairness Index: {calculate_jains_fairness_index(loads):.4f}")
    
    # Test validation
    print("\n4. Testing validation functions:")
    try:
        result = validate_positive_int(5, "test_value", min_val=1, max_val=10)
        print(f"   Valid int: {result}")
    except ValueError as e:
        print(f"   Error: {e}")
    
    try:
        result = validate_algorithm("Round Robin")
        print(f"   Valid algorithm: {result}")
    except ValueError as e:
        print(f"   Error: {e}")
    
    # Test color generation
    print("\n5. Testing color utilities:")
    for pid in range(5):
        color = generate_process_color(pid)
        print(f"   P{pid}: {color}")
    
    # Test data exporter
    print("\n6. Testing DataExporter:")
    exporter = DataExporter(output_dir="test_output")
    
    test_data = [
        {"name": "Process 1", "value": 10},
        {"name": "Process 2", "value": 20}
    ]
    filepath = exporter.export_to_csv(test_data, "test_export")
    print(f"   Exported to: {filepath}")
    
    # Cleanup test output
    import shutil
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")
        print("   Cleaned up test output")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
