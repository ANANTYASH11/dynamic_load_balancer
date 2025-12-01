"""
Processor Module for Dynamic Load Balancing Simulator

This module defines the Processor class which represents a CPU/core in the
multiprocessor system. Each processor has its own ready queue and executes
processes according to a scheduling algorithm.

Key OS Concepts Demonstrated:
- CPU Scheduling: Selecting which process to execute next
- Ready Queue: Queue of processes waiting for CPU time
- Context Switching: Changing from one process to another
- CPU Utilization: Measuring how busy the processor is

Author: Student
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Deque
from collections import deque
import threading
import time
import logging

from config import (
    ProcessState,
    SimulationConfig,
    DEFAULT_SIMULATION_CONFIG,
    GUIConfig,
    DEFAULT_GUI_CONFIG
)
from process import Process


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ProcessorStatistics:
    """
    Statistics tracking for a single processor.
    
    These metrics are essential for evaluating load balancing effectiveness:
    - High utilization = processor is being well-used
    - Low idle time = good workload distribution
    - Migration counts help evaluate algorithm efficiency
    """
    total_execution_time: int = 0      # Time spent executing processes
    total_idle_time: int = 0           # Time spent idle (no processes)
    total_context_switches: int = 0    # Number of process switches
    processes_completed: int = 0       # Number of processes finished
    processes_received: int = 0        # Total processes assigned
    processes_migrated_in: int = 0     # Processes received via migration
    processes_migrated_out: int = 0    # Processes sent via migration
    
    def get_utilization(self, total_time: int) -> float:
        """
        Calculate CPU utilization percentage.
        
        Utilization = (Execution Time / Total Time) × 100
        
        Args:
            total_time: Total simulation time elapsed
            
        Returns:
            Utilization as a fraction (0.0 to 1.0)
        """
        if total_time <= 0:
            return 0.0
        return min(1.0, self.total_execution_time / total_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'total_execution_time': self.total_execution_time,
            'total_idle_time': self.total_idle_time,
            'total_context_switches': self.total_context_switches,
            'processes_completed': self.processes_completed,
            'processes_received': self.processes_received,
            'processes_migrated_in': self.processes_migrated_in,
            'processes_migrated_out': self.processes_migrated_out
        }


class Processor:
    """
    Represents a CPU/processor in the multiprocessor system.
    
    In real operating systems, each CPU core:
    - Has its own set of registers and cache
    - Maintains a ready queue of processes
    - Executes processes according to a scheduling algorithm
    - Can be interrupted for context switches
    
    This simulation models:
    - Ready queue management (FIFO with priority support)
    - Process execution with time quantum
    - Load calculation for load balancing decisions
    - Statistics tracking for performance analysis
    
    Attributes:
        processor_id (int): Unique identifier for this processor
        ready_queue (Deque[Process]): Queue of processes waiting to execute
        current_process (Optional[Process]): Currently executing process
        statistics (ProcessorStatistics): Performance metrics
        is_active (bool): Whether processor is active in simulation
    """
    
    def __init__(
        self,
        processor_id: int,
        config: SimulationConfig = None,
        gui_config: GUIConfig = None
    ):
        """
        Initialize a new processor.
        
        Args:
            processor_id: Unique ID for this processor (0-indexed)
            config: Simulation configuration
            gui_config: GUI configuration for colors
        """
        self.processor_id = processor_id
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self.gui_config = gui_config or DEFAULT_GUI_CONFIG
        
        # Process management
        self.ready_queue: Deque[Process] = deque()
        self.current_process: Optional[Process] = None
        
        # Time tracking for current process (for time quantum)
        self._current_process_time: int = 0
        
        # Statistics
        self.statistics = ProcessorStatistics()
        
        # State
        self.is_active: bool = True
        
        # Thread safety for concurrent access
        self._lock = threading.Lock()
        
        # Execution history for visualization
        # Each entry: {'pid': int, 'start': int, 'end': int}
        self.execution_history: List[Dict[str, int]] = []
        self._last_execution_start: Optional[int] = None
        
        logger.debug(f"Processor {processor_id} initialized")
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Processor[ID={self.processor_id}, "
            f"Queue={len(self.ready_queue)}, "
            f"Current={self.current_process.pid if self.current_process else 'None'}, "
            f"Load={self.get_load():.2f}]"
        )
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"Processor(id={self.processor_id}, "
            f"queue_size={len(self.ready_queue)}, "
            f"current_pid={self.current_process.pid if self.current_process else None}, "
            f"stats={self.statistics.to_dict()})"
        )
    
    # =========================================================================
    # Process Queue Management
    # =========================================================================
    
    def add_process(self, process: Process, is_migration: bool = False) -> bool:
        """
        Add a process to this processor's ready queue.
        
        In OS terms: A new process has been assigned to this CPU's scheduler.
        
        Args:
            process: Process to add
            is_migration: True if process is being migrated from another processor
            
        Returns:
            True if process was added successfully
        """
        with self._lock:
            # Assign process to this processor
            process.assign_to_processor(self.processor_id)
            process.set_ready()
            
            # Add to ready queue
            self.ready_queue.append(process)
            
            # Update statistics
            self.statistics.processes_received += 1
            if is_migration:
                self.statistics.processes_migrated_in += 1
            
            logger.info(
                f"Process P{process.pid} added to Processor {self.processor_id} "
                f"(migration={is_migration}, queue_size={len(self.ready_queue)})"
            )
            
            return True
    
    def remove_process(self, process: Process) -> bool:
        """
        Remove a process from this processor (for migration).
        
        Args:
            process: Process to remove
            
        Returns:
            True if process was found and removed
        """
        with self._lock:
            # Check if it's the current process
            if self.current_process and self.current_process.pid == process.pid:
                # Preempt current process
                self._preempt_current(time.time())  # Will be updated with proper time
                return True
            
            # Search in ready queue
            try:
                self.ready_queue.remove(process)
                self.statistics.processes_migrated_out += 1
                logger.info(
                    f"Process P{process.pid} removed from Processor {self.processor_id} for migration"
                )
                return True
            except ValueError:
                logger.warning(
                    f"Process P{process.pid} not found in Processor {self.processor_id}"
                )
                return False
    
    def get_queue_size(self) -> int:
        """
        Get the number of processes in the ready queue.
        
        Returns:
            Number of waiting processes
        """
        return len(self.ready_queue)
    
    def get_all_processes(self) -> List[Process]:
        """
        Get all processes assigned to this processor.
        
        Returns:
            List of all processes (current + queued)
        """
        with self._lock:
            processes = list(self.ready_queue)
            if self.current_process:
                processes.insert(0, self.current_process)
            return processes
    
    def get_migratable_processes(self) -> List[Process]:
        """
        Get processes that can be migrated to another processor.
        
        Only processes in the ready queue (not currently running) can be migrated.
        This prevents interrupting active execution.
        
        Returns:
            List of processes that can be migrated
        """
        with self._lock:
            # Return copy of ready queue (not current process)
            return list(self.ready_queue)
    
    def has_processes(self) -> bool:
        """
        Check if processor has any processes.
        
        Returns:
            True if there are processes to execute
        """
        return self.current_process is not None or len(self.ready_queue) > 0
    
    # =========================================================================
    # Load Calculation
    # =========================================================================
    
    def get_load(self) -> float:
        """
        Calculate the current load of this processor.
        
        Load is calculated as a combination of:
        - Number of processes in queue
        - Remaining work (sum of remaining burst times)
        
        This provides a more accurate load measure than just queue length,
        as it considers how much work is actually waiting.
        
        Returns:
            Load value (higher = more loaded)
        """
        with self._lock:
            # Calculate total remaining work
            total_remaining = sum(p.remaining_time for p in self.ready_queue)
            
            if self.current_process:
                total_remaining += self.current_process.remaining_time
            
            # Normalize by some factor to get reasonable values
            # Load = (queue_size * weight) + (remaining_work / normalization)
            queue_weight = len(self.ready_queue) + (1 if self.current_process else 0)
            
            # Combine queue size and work remaining
            # This formula balances both factors
            load = queue_weight * 0.3 + total_remaining * 0.1
            
            return load
    
    def get_queue_load(self) -> int:
        """
        Get simple queue-based load (number of processes).
        
        Returns:
            Number of processes (current + queued)
        """
        count = len(self.ready_queue)
        if self.current_process:
            count += 1
        return count
    
    def get_work_remaining(self) -> int:
        """
        Get total remaining work (sum of remaining burst times).
        
        Returns:
            Total remaining time units
        """
        with self._lock:
            total = sum(p.remaining_time for p in self.ready_queue)
            if self.current_process:
                total += self.current_process.remaining_time
            return total
    
    def get_utilization(self, total_time: int) -> float:
        """
        Get CPU utilization percentage.
        
        Args:
            total_time: Total simulation time elapsed
            
        Returns:
            Utilization as fraction (0.0 to 1.0)
        """
        return self.statistics.get_utilization(total_time)
    
    def is_idle(self) -> bool:
        """
        Check if processor is idle (no work to do).
        
        Returns:
            True if no processes assigned
        """
        return self.current_process is None and len(self.ready_queue) == 0
    
    # =========================================================================
    # Process Execution
    # =========================================================================
    
    def execute_time_slice(self, current_time: int, time_quantum: int = None) -> Dict[str, Any]:
        """
        Execute one time slice of the scheduling algorithm.
        
        This is the main execution method called by the simulation loop.
        It handles:
        1. Selecting a process to run (if none running)
        2. Executing the process for up to time_quantum units
        3. Handling process completion or preemption
        
        Args:
            current_time: Current simulation time
            time_quantum: Time units to execute (uses config default if None)
            
        Returns:
            Dictionary with execution results:
            - 'executed': bool - whether execution occurred
            - 'process': Process or None - the process that ran
            - 'time_executed': int - actual time units executed
            - 'completed': bool - whether process finished
            - 'preempted': bool - whether process was preempted
        """
        if time_quantum is None:
            time_quantum = self.config.time_quantum
        
        result = {
            'executed': False,
            'process': None,
            'time_executed': 0,
            'completed': False,
            'preempted': False
        }
        
        with self._lock:
            # If no current process, try to get one from queue
            if self.current_process is None:
                if len(self.ready_queue) > 0:
                    self._select_next_process(current_time)
                else:
                    # Processor is idle
                    self.statistics.total_idle_time += 1
                    return result
            
            # Execute current process
            if self.current_process:
                process = self.current_process
                
                # Ensure process is in RUNNING state
                if process.state != ProcessState.RUNNING:
                    process.set_running(current_time)
                    self._last_execution_start = current_time
                
                # Calculate execution time
                exec_time = min(time_quantum, process.remaining_time)
                
                # Execute
                actual_executed = process.execute(exec_time, current_time)
                
                # Update statistics
                self.statistics.total_execution_time += actual_executed
                self._current_process_time += actual_executed
                
                result['executed'] = True
                result['process'] = process
                result['time_executed'] = actual_executed
                
                # Check if process completed
                if process.remaining_time <= 0:
                    self._complete_current_process(current_time + actual_executed)
                    result['completed'] = True
                    
                # Check if time quantum expired (preemption)
                elif self._current_process_time >= time_quantum:
                    self._preempt_current(current_time + actual_executed)
                    result['preempted'] = True
        
        return result
    
    def _select_next_process(self, current_time: int) -> Optional[Process]:
        """
        Select the next process from the ready queue.
        
        Uses FIFO ordering (processes are executed in arrival order).
        Priority can be considered by sorting the queue.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Selected process or None if queue empty
        """
        if len(self.ready_queue) == 0:
            return None
        
        # Get next process (FIFO)
        self.current_process = self.ready_queue.popleft()
        self.current_process.set_running(current_time)
        self._current_process_time = 0
        self._last_execution_start = current_time
        
        self.statistics.total_context_switches += 1
        
        logger.debug(
            f"Processor {self.processor_id}: Selected P{self.current_process.pid} for execution"
        )
        
        return self.current_process
    
    def _preempt_current(self, current_time: int) -> None:
        """
        Preempt the currently running process.
        
        The process is moved back to the ready queue (Round Robin style).
        
        Args:
            current_time: Current simulation time
        """
        if self.current_process is None:
            return
        
        process = self.current_process
        
        # Record execution segment
        if self._last_execution_start is not None:
            self.execution_history.append({
                'pid': process.pid,
                'start': self._last_execution_start,
                'end': current_time
            })
        
        # Preempt process
        process.preempt(current_time)
        
        # Add back to queue
        self.ready_queue.append(process)
        
        # Clear current
        self.current_process = None
        self._current_process_time = 0
        self._last_execution_start = None
        
        logger.debug(
            f"Processor {self.processor_id}: Preempted P{process.pid}, "
            f"remaining={process.remaining_time}"
        )
    
    def _complete_current_process(self, current_time: int) -> None:
        """
        Mark the current process as completed.
        
        Args:
            current_time: Current simulation time
        """
        if self.current_process is None:
            return
        
        process = self.current_process
        
        # Record execution segment
        if self._last_execution_start is not None:
            self.execution_history.append({
                'pid': process.pid,
                'start': self._last_execution_start,
                'end': current_time
            })
        
        # Complete process
        process.set_completed(current_time)
        
        # Update statistics
        self.statistics.processes_completed += 1
        
        logger.info(
            f"Processor {self.processor_id}: Completed P{process.pid} at time {current_time}"
        )
        
        # Clear current
        self.current_process = None
        self._current_process_time = 0
        self._last_execution_start = None
    
    def update_waiting_times(self, time_units: int = 1) -> None:
        """
        Update waiting times for all processes in the ready queue.
        
        Called each simulation tick to track how long processes wait.
        
        Args:
            time_units: Time units elapsed
        """
        with self._lock:
            for process in self.ready_queue:
                process.update_waiting_time(time_units)
    
    # =========================================================================
    # Statistics and Information
    # =========================================================================
    
    def get_statistics(self) -> ProcessorStatistics:
        """
        Get the processor's statistics.
        
        Returns:
            ProcessorStatistics object
        """
        return self.statistics
    
    def get_color(self, total_time: int) -> str:
        """
        Get the display color based on current utilization.
        
        Args:
            total_time: Total simulation time
            
        Returns:
            Hex color string
        """
        utilization = self.get_utilization(total_time)
        return self.gui_config.get_load_color(utilization)
    
    def reset(self) -> None:
        """
        Reset the processor to initial state.
        
        Used when starting a new simulation run.
        """
        with self._lock:
            self.ready_queue.clear()
            self.current_process = None
            self._current_process_time = 0
            self.statistics = ProcessorStatistics()
            self.execution_history.clear()
            self._last_execution_start = None
            self.is_active = True
        
        logger.debug(f"Processor {self.processor_id} reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert processor state to dictionary for logging/serialization.
        
        Returns:
            Dictionary with processor information
        """
        with self._lock:
            return {
                'processor_id': self.processor_id,
                'queue_size': len(self.ready_queue),
                'current_process_pid': self.current_process.pid if self.current_process else None,
                'load': self.get_load(),
                'work_remaining': self.get_work_remaining(),
                'is_idle': self.is_idle(),
                'statistics': self.statistics.to_dict(),
                'queue_pids': [p.pid for p in self.ready_queue]
            }


# =============================================================================
# PROCESSOR MANAGER
# =============================================================================

class ProcessorManager:
    """
    Manages a collection of processors in the multiprocessor system.
    
    This class provides:
    - Creation and management of multiple processors
    - Aggregate statistics across all processors
    - Helper methods for load balancing decisions
    
    In real OS terms, this would be similar to the functionality provided
    by the kernel's CPU management subsystem.
    """
    
    def __init__(self, num_processors: int = 4, config: SimulationConfig = None):
        """
        Initialize the processor manager.
        
        Args:
            num_processors: Number of processors to create
            config: Simulation configuration
        """
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self.processors: List[Processor] = []
        
        # Create processors
        for i in range(num_processors):
            processor = Processor(processor_id=i, config=self.config)
            self.processors.append(processor)
        
        logger.info(f"ProcessorManager initialized with {num_processors} processors")
    
    def __len__(self) -> int:
        """Return number of processors."""
        return len(self.processors)
    
    def __getitem__(self, index: int) -> Processor:
        """Get processor by index."""
        return self.processors[index]
    
    def __iter__(self):
        """Iterate over processors."""
        return iter(self.processors)
    
    # =========================================================================
    # Processor Access
    # =========================================================================
    
    def get_processor(self, processor_id: int) -> Optional[Processor]:
        """
        Get a processor by ID.
        
        Args:
            processor_id: Processor ID to look up
            
        Returns:
            Processor object or None if not found
        """
        if 0 <= processor_id < len(self.processors):
            return self.processors[processor_id]
        return None
    
    def get_all_processors(self) -> List[Processor]:
        """
        Get all processors.
        
        Returns:
            List of all Processor objects
        """
        return self.processors.copy()
    
    # =========================================================================
    # Load Information
    # =========================================================================
    
    def get_loads(self) -> List[float]:
        """
        Get load values for all processors.
        
        Returns:
            List of load values
        """
        return [p.get_load() for p in self.processors]
    
    def get_queue_sizes(self) -> List[int]:
        """
        Get queue sizes for all processors.
        
        Returns:
            List of queue sizes
        """
        return [p.get_queue_size() for p in self.processors]
    
    def get_utilizations(self, total_time: int) -> List[float]:
        """
        Get utilization percentages for all processors.
        
        Args:
            total_time: Total simulation time
            
        Returns:
            List of utilizations (0.0 to 1.0)
        """
        return [p.get_utilization(total_time) for p in self.processors]
    
    def get_least_loaded_processor(self) -> Processor:
        """
        Find the processor with the lowest load.
        
        Used by Least Loaded First algorithm.
        
        Returns:
            Processor with minimum load
        """
        return min(self.processors, key=lambda p: p.get_load())
    
    def get_most_loaded_processor(self) -> Processor:
        """
        Find the processor with the highest load.
        
        Used by threshold-based algorithm for migration decisions.
        
        Returns:
            Processor with maximum load
        """
        return max(self.processors, key=lambda p: p.get_load())
    
    def get_load_variance(self) -> float:
        """
        Calculate the variance in loads across processors.
        
        High variance indicates unbalanced load distribution.
        
        Returns:
            Load variance value
        """
        loads = self.get_loads()
        if len(loads) == 0:
            return 0.0
        
        mean_load = sum(loads) / len(loads)
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        return variance
    
    def get_load_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive load statistics.
        
        Returns:
            Dictionary with min, max, mean, variance of loads
        """
        loads = self.get_loads()
        if len(loads) == 0:
            return {'min': 0, 'max': 0, 'mean': 0, 'variance': 0}
        
        mean_load = sum(loads) / len(loads)
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        
        return {
            'min': min(loads),
            'max': max(loads),
            'mean': mean_load,
            'variance': variance,
            'std_dev': variance ** 0.5
        }
    
    # =========================================================================
    # Aggregate Operations
    # =========================================================================
    
    def execute_all(self, current_time: int) -> List[Dict[str, Any]]:
        """
        Execute one time slice on all processors.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of execution results from each processor
        """
        results = []
        for processor in self.processors:
            result = processor.execute_time_slice(current_time)
            result['processor_id'] = processor.processor_id
            results.append(result)
        return results
    
    def update_all_waiting_times(self, time_units: int = 1) -> None:
        """
        Update waiting times for all processes on all processors.
        
        Args:
            time_units: Time units elapsed
        """
        for processor in self.processors:
            processor.update_waiting_times(time_units)
    
    def get_all_processes(self) -> List[Process]:
        """
        Get all processes across all processors.
        
        Returns:
            List of all processes
        """
        processes = []
        for processor in self.processors:
            processes.extend(processor.get_all_processes())
        return processes
    
    def get_completed_count(self) -> int:
        """
        Get total number of completed processes.
        
        Returns:
            Total completed processes across all processors
        """
        return sum(p.statistics.processes_completed for p in self.processors)
    
    def get_total_processes(self) -> int:
        """
        Get total number of processes received.
        
        Returns:
            Total processes assigned across all processors
        """
        return sum(p.statistics.processes_received for p in self.processors)
    
    def all_idle(self) -> bool:
        """
        Check if all processors are idle.
        
        Returns:
            True if all processors have no work
        """
        return all(p.is_idle() for p in self.processors)
    
    def reset_all(self) -> None:
        """
        Reset all processors to initial state.
        """
        for processor in self.processors:
            processor.reset()
        
        logger.info("All processors reset")
    
    def get_aggregate_statistics(self, total_time: int) -> Dict[str, Any]:
        """
        Get aggregate statistics across all processors.
        
        Args:
            total_time: Total simulation time
            
        Returns:
            Dictionary with aggregate metrics
        """
        total_execution = sum(p.statistics.total_execution_time for p in self.processors)
        total_idle = sum(p.statistics.total_idle_time for p in self.processors)
        total_switches = sum(p.statistics.total_context_switches for p in self.processors)
        total_completed = sum(p.statistics.processes_completed for p in self.processors)
        total_migrations_in = sum(p.statistics.processes_migrated_in for p in self.processors)
        
        utilizations = self.get_utilizations(total_time)
        
        return {
            'total_execution_time': total_execution,
            'total_idle_time': total_idle,
            'total_context_switches': total_switches,
            'total_completed': total_completed,
            'total_migrations': total_migrations_in,
            'average_utilization': sum(utilizations) / len(utilizations) if utilizations else 0,
            'min_utilization': min(utilizations) if utilizations else 0,
            'max_utilization': max(utilizations) if utilizations else 0,
            'load_variance': self.get_load_variance(),
            'per_processor': [p.to_dict() for p in self.processors]
        }


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    from process import Process, ProcessGenerator, ProcessPriority
    
    print("=" * 60)
    print("Processor Module Test")
    print("=" * 60)
    
    # Test single processor
    print("\n1. Testing single Processor:")
    processor = Processor(processor_id=0)
    print(f"   Created: {processor}")
    
    # Create test processes
    generator = ProcessGenerator()
    processes = generator.generate_predefined_test_set()[:4]
    
    print("\n2. Adding processes to processor:")
    for p in processes:
        processor.add_process(p)
        print(f"   Added: P{p.pid} (burst={p.burst_time})")
    
    print(f"\n   Queue size: {processor.get_queue_size()}")
    print(f"   Load: {processor.get_load():.2f}")
    print(f"   Work remaining: {processor.get_work_remaining()}")
    
    # Test execution
    print("\n3. Testing execution:")
    current_time = 0
    for _ in range(10):
        result = processor.execute_time_slice(current_time)
        if result['executed']:
            print(f"   Time {current_time}: Executed P{result['process'].pid}, "
                  f"time={result['time_executed']}, "
                  f"completed={result['completed']}, "
                  f"preempted={result['preempted']}")
        else:
            print(f"   Time {current_time}: Idle")
        current_time += 1
    
    print(f"\n   Completed: {processor.statistics.processes_completed}")
    print(f"   Utilization: {processor.get_utilization(current_time)*100:.1f}%")
    
    # Test ProcessorManager
    print("\n4. Testing ProcessorManager:")
    manager = ProcessorManager(num_processors=4)
    print(f"   Created {len(manager)} processors")
    
    # Add processes to different processors
    generator.reset()
    test_processes = generator.generate_processes(12)
    
    for i, p in enumerate(test_processes):
        processor_id = i % len(manager)
        manager[processor_id].add_process(p)
    
    print("\n   Loads after distribution:")
    for i, proc in enumerate(manager):
        print(f"   Processor {i}: Load={proc.get_load():.2f}, Queue={proc.get_queue_size()}")
    
    stats = manager.get_load_statistics()
    print(f"\n   Load Statistics:")
    print(f"   Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
    print(f"   Mean: {stats['mean']:.2f}, Variance: {stats['variance']:.2f}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
