import torch
import gc
import sys
import time
import os
from typing import Dict, List, Optional, Set, Tuple
from ase.units import Hartree, Bohr
import numpy as np
import matplotlib.pyplot as plt

class MemoryTracker:
    """
    A utility for tracking memory usage in PyTorch and identifying potential memory leaks.
    """
    
    def __init__(self, log_dir="./memory_logs", plot_memory=True):
        """
        Initialize the memory tracker.
        
        Args:
            log_dir: Directory to save memory logs and plots
            plot_memory: Whether to generate plots of memory usage
        """
        self.log_dir = log_dir
        self.plot_memory = plot_memory
        self.memory_stats = []
        self.checkpoint_timestamps = []
        self.tensor_counts = []
        self.tensor_sizes = []
        self.checkpoint_labels = []
        self.leaked_tensors = set()
        self.total_tensor_count = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tensor tracking
        self._tensor_creation_locations = {}
        
    def _sizeof_fmt(self, num, suffix='B'):
        """Format file sizes in human-readable format"""
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 1024.0:
                return f"{num:.2f} {unit}{suffix}"
            num /= 1024.0
        return f"{num:.2f} Y{suffix}"
        
    def checkpoint(self, label=""):
        """
        Record the current memory usage at a checkpoint.
        
        Args:
            label: A label for this checkpoint for easier identification
        """
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Get memory stats
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
        else:
            memory_allocated = 0
            memory_reserved = 0
            
        # Count tensors with gradients
        grad_tensor_count = 0
        grad_tensor_memory = 0
        requires_grad_tensor_count = 0
        requires_grad_tensor_memory = 0
        
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    self.total_tensor_count += 1
                    if obj.grad is not None:
                        grad_tensor_count += 1
                        grad_tensor_memory += obj.element_size() * obj.nelement()
                    if obj.requires_grad:
                        requires_grad_tensor_count += 1
                        requires_grad_tensor_memory += obj.element_size() * obj.nelement()
            except:
                pass
                
        # Record stats
        timestamp = time.time()
        self.memory_stats.append({
            'timestamp': timestamp,
            'memory_allocated': memory_allocated,
            'memory_reserved': memory_reserved,
            'grad_tensor_count': grad_tensor_count,
            'grad_tensor_memory': grad_tensor_memory,
            'requires_grad_tensor_count': requires_grad_tensor_count,
            'requires_grad_tensor_memory': requires_grad_tensor_memory
        })
        
        self.checkpoint_timestamps.append(timestamp)
        self.checkpoint_labels.append(label)
        
        # Detailed tensor counting and tracking
        self._count_tensors_by_type()
        
        # Print current memory usage
        print(f"Checkpoint: {label}")
        print(f"  Allocated: {self._sizeof_fmt(memory_allocated)}")
        print(f"  Reserved:  {self._sizeof_fmt(memory_reserved)}")
        print(f"  Tensors with grad: {grad_tensor_count} ({self._sizeof_fmt(grad_tensor_memory)})")
        print(f"  Tensors requiring grad: {requires_grad_tensor_count} ({self._sizeof_fmt(requires_grad_tensor_memory)})")
        
    def _count_tensors_by_type(self):
        """Count tensors of each dtype and shape, to track patterns"""
        tensor_counts_by_type = {}
        tensor_total_size_by_type = {}
        
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    # Skip tensors that are part of the autograd engine
                    if obj.is_leaf:
                        key = (str(obj.dtype), str(tuple(obj.shape)), bool(obj.requires_grad))
                        
                        if key not in tensor_counts_by_type:
                            tensor_counts_by_type[key] = 0
                            tensor_total_size_by_type[key] = 0
                            
                        tensor_counts_by_type[key] += 1
                        tensor_total_size_by_type[key] += obj.element_size() * obj.nelement()
            except:
                pass
                
        self.tensor_counts.append(tensor_counts_by_type)
        self.tensor_sizes.append(tensor_total_size_by_type)
        
    def dump_tensor_report(self, top_n=20):
        """
        Generate a report of the top N tensor types by memory usage.
        
        Args:
            top_n: Number of top tensor types to show
        """
        if not self.tensor_sizes:
            print("No tensor data collected")
            return
            
        # Get the latest tensor size data
        latest_sizes = self.tensor_sizes[-1]
        
        # Sort by size
        sorted_types = sorted(latest_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Print report
        print(f"Top {top_n} tensor types by memory usage:")
        print(f"{'Type':<50} {'Count':<10} {'Total Size':<15} {'Requires Grad':<15}")
        print("-" * 90)
        
        for i, ((dtype, shape, req_grad), size) in enumerate(sorted_types[:top_n]):
            count = self.tensor_counts[-1].get((dtype, shape, req_grad), 0)
            print(f"{dtype}, {shape:<40} {count:<10} {self._sizeof_fmt(size):<15} {req_grad}")
            
    def track_tensor_allocations(self, enable=True):
        """
        Enable tracking of tensor creation locations to help identify leak sources.
        This has significant performance overhead and should only be used for debugging.
        
        Args:
            enable: Whether to enable tracking
        """
        def tensor_allocation_tracker(tensor):
            import traceback
            self._tensor_creation_locations[id(tensor)] = traceback.extract_stack()
            return tensor
            
        if enable:
            torch.Tensor.__old_new__ = torch.Tensor.__new__
            torch.Tensor.__new__ = tensor_allocation_tracker
        else:
            if hasattr(torch.Tensor, '__old_new__'):
                torch.Tensor.__new__ = torch.Tensor.__old_new__
                
    def find_leaked_tensors(self, min_lifetime=3, print_details=True):
        """
        Identify tensors that have persisted through multiple checkpoints.

        Args:
            min_lifetime: Minimum number of checkpoints a tensor must persist through to be considered a leak
            print_details: Whether to print details about the leaked tensors
        """
        current_tensors = set()
        leaked_info = []

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_leaf:
                    tensor_id = id(obj)
                    current_tensors.add(tensor_id)

                    if tensor_id in self.leaked_tensors:
                        # Already tracked this tensor
                        continue

                    if hasattr(obj, '_tracker_checkpoint_count'):
                        obj._tracker_checkpoint_count += 1
                        if obj._tracker_checkpoint_count >= min_lifetime:
                            self.leaked_tensors.add(tensor_id)
                            if print_details:
                                # Store detailed info about the leaked tensor
                                leaked_info.append({
                                    'id': tensor_id,
                                    'shape': tuple(obj.shape),
                                    'size_bytes': obj.element_size() * obj.nelement(),
                                    'dtype': str(obj.dtype),
                                    'device': str(obj.device),
                                    'requires_grad': obj.requires_grad
                                })
                    else:
                        obj._tracker_checkpoint_count = 1
            except:
                pass
            
        # Print newly identified leaked tensors with details
        if self.leaked_tensors and print_details:
            print(f"\nFound {len(self.leaked_tensors)} potential leaked tensors")

            if leaked_info:
                # Sort by size
                leaked_info.sort(key=lambda x: x['size_bytes'], reverse=True)

                # Print top 20 largest tensors
                print("\nTop 20 largest leaked tensors:")
                print(f"{'Shape':<20} {'Size':<15} {'Dtype':<10} {'Requires Grad':<15}")
                print("-" * 70)

                for i, info in enumerate(leaked_info[:20]):
                    print(f"{str(info['shape']):<20} {self._sizeof_fmt(info['size_bytes']):<15} "
                          f"{info['dtype']:<10} {str(info['requires_grad']):<15}")
            
    def plot_memory_usage(self):
        """Generate a plot of memory usage over time"""
        if not self.memory_stats or not self.plot_memory:
            return
            
        # Extract data for plotting
        timestamps = [s['timestamp'] - self.memory_stats[0]['timestamp'] for s in self.memory_stats]
        allocated = [s['memory_allocated'] / (1024 * 1024) for s in self.memory_stats]  # Convert to MB
        reserved = [s['memory_reserved'] / (1024 * 1024) for s in self.memory_stats]  # Convert to MB
        grad_counts = [s['grad_tensor_count'] for s in self.memory_stats]
        requires_grad_counts = [s['requires_grad_tensor_count'] for s in self.memory_stats]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Memory plot
        ax1.plot(timestamps, allocated, 'b-', label='Allocated')
        ax1.plot(timestamps, reserved, 'r-', label='Reserved')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title('PyTorch Memory Usage Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Tensor count plot
        ax2.plot(timestamps, grad_counts, 'g-', label='Tensors with grad')
        ax2.plot(timestamps, requires_grad_counts, 'm-', label='Tensors requiring grad')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Tensor Count')
        ax2.set_title('Tensor Counts Over Time')
        ax2.legend()
        ax2.grid(True)
        
        # Set x-axis ticks to be more evenly spaced
        # Choose at most 10-15 ticks regardless of how many checkpoints we have
        n_ticks = min(15, len(timestamps))
        if len(timestamps) > n_ticks:
            tick_indices = np.linspace(0, len(timestamps)-1, n_ticks, dtype=int)
            tick_positions = [timestamps[i] for i in tick_indices]
            
            # Use step numbers as tick labels if available
            if len(self.checkpoint_labels) >= len(timestamps):
                tick_labels = [self.checkpoint_labels[i].replace('Step ', '') for i in tick_indices]
                # If not all labels have 'Step' in them, use timestamps
                if not all('Step' in label for label in self.checkpoint_labels):
                    tick_labels = [f"{t:.1f}s" for t in tick_positions]
            else:
                tick_labels = [f"{t:.1f}s" for t in tick_positions]
                
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels)
        
        # Add reference lines for significant checkpoints but limit the number
        important_checkpoints = []
        if len(self.checkpoint_labels) > 0:
            # Find checkpoints with meaningful labels (not just step numbers)
            # We also add the first, last, and some evenly spaced ones in the middle
            for i, label in enumerate(self.checkpoint_labels):
                if i == 0 or i == len(self.checkpoint_labels)-1 or not label.startswith('Step '):
                    important_checkpoints.append(i)
        
            # If we still have too few important checkpoints, add some evenly spaced ones
            if len(important_checkpoints) < 5 and len(self.checkpoint_labels) > 10:
                additional_indices = np.linspace(0, len(self.checkpoint_labels)-1, 5, dtype=int)
                important_checkpoints.extend(additional_indices)
                important_checkpoints = sorted(list(set(important_checkpoints)))
        
            # Add vertical lines for important checkpoints
            for i in important_checkpoints:
                if i < len(self.checkpoint_timestamps):
                    t = self.checkpoint_timestamps[i] - self.memory_stats[0]['timestamp']
                    label = self.checkpoint_labels[i] if i < len(self.checkpoint_labels) else ""
                    if label:  # Only add labels that aren't empty
                        ax1.axvline(x=t, color='k', linestyle='--', alpha=0.3)
                        ax2.axvline(x=t, color='k', linestyle='--', alpha=0.3)
                        # Add the label at the top of the upper plot
                        ax1.text(t, max(allocated) * 1.05, label, rotation=45, ha='right')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'memory_usage.png'))
        plt.close(fig)
        
    def check_computational_graph(self, sample_tensor=None):
        """
        Analyze the computational graph to find potential issues.
        
        Args:
            sample_tensor: A tensor that's part of the computational graph to analyze
        """
        if sample_tensor is None:
            # Try to find a tensor with grad
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.grad is not None:
                        sample_tensor = obj
                        break
                except:
                    pass
                    
        if sample_tensor is None:
            print("No tensor with grad found to analyze")
            return
            
        # Print grad function chain
        print("Gradient function chain:")
        node = sample_tensor.grad_fn
        
        if node is None:
            print("  No grad_fn (leaf tensor)")
            return
            
        depth = 0
        while node is not None:
            print(f"  {depth}: {type(node).__name__}")
            if hasattr(node, 'next_functions'):
                node = node.next_functions[0][0] if node.next_functions else None
            else:
                node = None
            depth += 1
            if depth > 100:
                print("  ... (graph too deep, truncating)")
                break
                
    def enable_anomaly_detection(self):
        """Enable PyTorch's anomaly detection for identifying backward pass issues"""
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled - this will significantly slow down your code")
        
    def clean_memory(self):
        """
        Attempt to clean up memory by forcing garbage collection and emptying the CUDA cache
        """
        # Record memory before cleaning
        if torch.cuda.is_available():
            before_allocated = torch.cuda.memory_allocated()
            before_reserved = torch.cuda.memory_reserved()
        else:
            before_allocated = 0
            before_reserved = 0
            
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Record memory after cleaning
        if torch.cuda.is_available():
            after_allocated = torch.cuda.memory_allocated()
            after_reserved = torch.cuda.memory_reserved()
        else:
            after_allocated = 0
            after_reserved = 0
            
        # Print results
        print(f"Memory cleanup:")
        print(f"  Allocated: {self._sizeof_fmt(before_allocated)} -> {self._sizeof_fmt(after_allocated)}")
        print(f"  Reserved:  {self._sizeof_fmt(before_reserved)} -> {self._sizeof_fmt(after_reserved)}")
        print(f"  Freed:     {self._sizeof_fmt(before_allocated - after_allocated)}")
        
    def find_unused_parameters(self):
        """Find parameters that might not be connected to the graph"""
        if torch.cuda.is_available():
            # We need to analyze tensors from the most recent computation
            print("Analyzing tensors that might be disconnected from the computational graph...")
            count = 0
            size = 0
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.requires_grad and obj.grad is None and obj.is_leaf:
                        count += 1
                        size += obj.element_size() * obj.nelement()
                        print(f"  Tensor with shape {obj.shape}, dtype {obj.dtype} requires_grad but has no gradient")
                except:
                    pass
            print(f"Found {count} tensors requiring grad but with no gradient ({self._sizeof_fmt(size)})")

    def finalize(self, skip_plots=False, limit_tensor_report=True):
        """
        Finalize the memory tracking, generate plots and reports,
        and perform final analysis.

        Args:
            skip_plots: If True, skip plot generation to save time
            limit_tensor_report: If True, limit the tensor analysis to save time
        """
        self.checkpoint("End of tracking")

        if self.plot_memory and not skip_plots:
            self.plot_memory_usage()

        # Generate summary report
        with open(os.path.join(self.log_dir, 'memory_report.txt'), 'w') as f:
            # Redirect output to file
            old_stdout = sys.stdout
            sys.stdout = f

            print("=" * 80)
            print("MEMORY TRACKING REPORT")
            print("=" * 80)
            print(f"Tracked {len(self.memory_stats)} checkpoints")

            # Print checkpoint stats
            print("\nCHECKPOINTS:")
            print("-" * 80)
            for i, stats in enumerate(self.memory_stats):
                label = self.checkpoint_labels[i] if i < len(self.checkpoint_labels) else ""
                print(f"Checkpoint {i} - {label}")
                print(f"  Allocated: {self._sizeof_fmt(stats['memory_allocated'])}")
                print(f"  Reserved:  {self._sizeof_fmt(stats['memory_reserved'])}")
                print(f"  Tensors with grad: {stats['grad_tensor_count']} ({self._sizeof_fmt(stats['grad_tensor_memory'])})")
                print(f"  Tensors requiring grad: {stats['requires_grad_tensor_count']} ({self._sizeof_fmt(stats['requires_grad_tensor_memory'])})")

            # Print tensor type analysis from the last checkpoint
            if not limit_tensor_report:
                print("\nTENSOR TYPE ANALYSIS (LAST CHECKPOINT):")
                print("-" * 80)
                self.dump_tensor_report(top_n=30)
            else:
                print("\nTENSOR TYPE ANALYSIS (LIMITED):")
                print("-" * 80)
                print("Tensor analysis skipped to save time")

            # Print leaked tensors analysis
            print("\nPOTENTIAL MEMORY LEAKS:")
            print("-" * 80)
            print(f"Found {len(self.leaked_tensors)} potential leaked tensors")

            # Restore stdout
            sys.stdout = old_stdout

        # Print summary
        print(f"Memory tracking complete. Reports saved to {self.log_dir}")
        print(f"See {os.path.join(self.log_dir, 'memory_report.txt')} for detailed analysis")


def add_memory_tracking_to_md(cmm_ase, memory_tracker, check_interval=10):
    """
    Patch the CMM_ASE calculator to add memory tracking at regular intervals.
    
    Args:
        cmm_ase: The CMM_ASE calculator instance
        memory_tracker: A MemoryTracker instance
        check_interval: How often to check memory (in MD steps)
    """
    # Save the original calculate method
    original_calculate = cmm_ase.calculate
    
    # Step counter
    step_counter = [0]  # Use a list for nonlocal behavior
    
    # Create patched method
    def calculate_with_memory_tracking(atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        # Call original method
        result = original_calculate(atoms, properties, system_changes)
        
        # Check for memory tracking
        step_counter[0] += 1
        if step_counter[0] % check_interval == 0:
            memory_tracker.checkpoint(f"Step {step_counter[0]}")
            
            # Every 100 steps, do a more detailed check
            if step_counter[0] % (check_interval * 10) == 0:
                memory_tracker.find_leaked_tensors()
                memory_tracker.find_unused_parameters()
                memory_tracker.clean_memory()
                
        return result
    
    # Apply the patch
    cmm_ase.calculate = calculate_with_memory_tracking
    
    return cmm_ase

class BackwardMemoryMonitor:
    """
    A lightweight monitor for tracking memory usage during backward passes.
    This avoids complex patching or weak reference issues.
    """
    
    def __init__(self):
        self.backward_memory_stats = []
        self.backward_count = 0
    
    def _sizeof_fmt(self, num, suffix='B'):
        """Format file sizes in human-readable format"""
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 1024.0:
                return f"{num:.2f} {unit}{suffix}"
            num /= 1024.0
        return f"{num:.2f} Y{suffix}"
    
    def monitor_backward(self, tensor, *args, **kwargs):
        """
        Monitor memory usage during a backward pass.
        
        Args:
            tensor: The tensor to backward through
            *args, **kwargs: Arguments to pass to tensor.backward()
        """
        # Get pre-backward memory stats
        if torch.cuda.is_available():
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            pre_mem_allocated = torch.cuda.memory_allocated()
            pre_mem_reserved = torch.cuda.memory_reserved()
        else:
            pre_mem_allocated = 0
            pre_mem_reserved = 0
        
        pre_time = time.time()
        
        # Count tensors before backward
        pre_tensor_count = self._count_tensors_requiring_grad()
        
        # Print pre-backward info
        self.backward_count += 1
        print(f"\n==== Backward Pass #{self.backward_count} ====")
        print(f"Pre-backward allocated memory: {self._sizeof_fmt(pre_mem_allocated)}")
        print(f"Pre-backward tensors requiring grad: {pre_tensor_count}")
        print(f"Tensor shape: {tensor.shape}, requires_grad: {tensor.requires_grad}")
        
        # Perform backward
        tensor.backward(*args, **kwargs)
        
        # Get post-backward memory stats
        post_time = time.time()
        
        if torch.cuda.is_available():
            post_mem_allocated = torch.cuda.memory_allocated()
            post_mem_reserved = torch.cuda.memory_reserved()
            peak_mem_allocated = torch.cuda.max_memory_allocated()
            peak_mem_reserved = torch.cuda.max_memory_reserved()
        else:
            post_mem_allocated = 0
            post_mem_reserved = 0
            peak_mem_allocated = 0
            peak_mem_reserved = 0
        
        # Count tensors after backward
        post_tensor_count = self._count_tensors_requiring_grad()
        
        # Record stats
        duration_ms = (post_time - pre_time) * 1000
        memory_diff = {
            'pre_allocated': pre_mem_allocated,
            'post_allocated': post_mem_allocated,
            'diff_allocated': post_mem_allocated - pre_mem_allocated,
            'pre_reserved': pre_mem_reserved,
            'post_reserved': post_mem_reserved,
            'diff_reserved': post_mem_reserved - pre_mem_reserved,
            'peak_allocated': peak_mem_allocated,
            'peak_reserved': peak_mem_reserved,
            'duration_ms': duration_ms,
            'pre_tensor_count': pre_tensor_count,
            'post_tensor_count': post_tensor_count,
        }
        
        self.backward_memory_stats.append(memory_diff)
        
        # Print post-backward info
        print(f"Post-backward allocated memory: {self._sizeof_fmt(post_mem_allocated)}")
        print(f"Peak memory during backward: {self._sizeof_fmt(peak_mem_allocated)}")
        print(f"Memory change: {self._sizeof_fmt(post_mem_allocated - pre_mem_allocated)}")
        print(f"Duration: {duration_ms:.2f} ms")
        print(f"Tensors requiring grad: {pre_tensor_count} → {post_tensor_count}")
        
        # If memory grew significantly, print warning
        if post_mem_allocated > pre_mem_allocated * 1.2:
            print("WARNING: Significant memory growth during backward pass!")
            print("Consider clearing gradients immediately after use.")
        
        # If peak memory was much higher than final allocated, there was a spike
        if peak_mem_allocated > post_mem_allocated * 1.5:
            print("WARNING: Large memory spike during backward computation!")
            print(f"  Peak: {self._sizeof_fmt(peak_mem_allocated)} vs Final: {self._sizeof_fmt(post_mem_allocated)}")
            print("This could indicate temporary tensors being created during gradients calculation.")
        
        # Find and print largest tensor
        largest_tensor = self._find_largest_tensor()
        if largest_tensor:
            print(f"Largest tensor: {largest_tensor['shape']} ({self._sizeof_fmt(largest_tensor['size'])})")
            print(f"  requires_grad: {largest_tensor['requires_grad']}, is_leaf: {largest_tensor['is_leaf']}")
        
        return memory_diff
    
    def _count_tensors_requiring_grad(self):
        """Count the number of tensors requiring grad"""
        count = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.requires_grad:
                    count += 1
            except:
                pass
        return count
    
    def _find_largest_tensor(self):
        """Find the largest tensor in memory"""
        largest_size = 0
        largest_info = None
        
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    size = obj.element_size() * obj.nelement()
                    if size > largest_size:
                        largest_size = size
                        largest_info = {
                            'shape': list(obj.shape),
                            'size': size,
                            'requires_grad': obj.requires_grad,
                            'is_leaf': obj.is_leaf if hasattr(obj, 'is_leaf') else None,
                            'dtype': str(obj.dtype)
                        }
            except:
                pass
        
        return largest_info
    
    def force_gc(self):
        """Force garbage collection and CUDA cache clearing"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Forced garbage collection")
    
    def summarize(self):
        """Print a summary of all backward passes"""
        if not self.backward_memory_stats:
            print("No backward passes recorded yet")
            return
        
        print("\n==== Backward Pass Memory Summary ====")
        print(f"Total backward passes: {len(self.backward_memory_stats)}")
        
        # Calculate average memory changes
        avg_alloc_diff = np.mean([stats['diff_allocated'] for stats in self.backward_memory_stats])
        max_alloc_diff = max([stats['diff_allocated'] for stats in self.backward_memory_stats])
        max_peak = max([stats['peak_allocated'] for stats in self.backward_memory_stats])
        avg_duration = np.mean([stats['duration_ms'] for stats in self.backward_memory_stats])
        
        print(f"Average memory change: {self._sizeof_fmt(avg_alloc_diff)}")
        print(f"Maximum memory change: {self._sizeof_fmt(max_alloc_diff)}")
        print(f"Maximum peak memory: {self._sizeof_fmt(max_peak)}")
        print(f"Average duration: {avg_duration:.2f} ms")
        
        # Check if memory is growing across backward passes
        first_allocated = self.backward_memory_stats[0]['pre_allocated']
        last_allocated = self.backward_memory_stats[-1]['post_allocated']
        
        if last_allocated > first_allocated * 1.2:
            print("\nWARNING: Memory is growing across backward passes!")
            print(f"First backward pre-allocated: {self._sizeof_fmt(first_allocated)}")
            print(f"Last backward post-allocated: {self._sizeof_fmt(last_allocated)}")
            print(f"Growth ratio: {last_allocated / first_allocated:.2f}x")
            
            # Most likely culprits based on common PyTorch patterns
            print("\nMost likely memory leak sources:")
            print("1. Gradients not being cleared after backward (use tensor.grad = None)")
            print("2. Tensors with requires_grad=True being stored in global lists/dictionaries")
            print("3. Saved tensors from backward pass not being released")
            print("4. Tensors from autograd history accumulating (use .detach() when storing results)")
        else:
            print("\nMemory appears stable across backward passes.")

def patch_cmm_ase_with_backward_memory_monitor(calculator):
    """
    Patch a CMM_ASE calculator with the SimpleBackwardMonitor.
    
    Args:
        calculator: The CMM_ASE calculator instance
        
    Returns:
        SimpleBackwardMonitor: The monitor instance
    """
    monitor = BackwardMemoryMonitor()
    
    # Save original _evaluate_ff method
    original_evaluate_ff = calculator._evaluate_ff
    
    # Create patched method
    def evaluate_ff_with_monitoring():
        # Call up to the point where backward would happen
        calculator._energies = calculator._ff.evaluate(calculator._cm, calculator._topology, calculator._params, reset_grads=True)
        
        # Use the monitor for backward
        print("\nMonitoring backward pass in _evaluate_ff...")
        monitor.monitor_backward(calculator._energies['total'])
        
        # Continue with the rest of the method
        calculator.results['energy'] = float(calculator._energies['total'].detach().cpu()) * Hartree
        calculator.results['forces'] = -calculator._cm.coords.grad.detach().cpu().numpy() * (Hartree / Bohr)
        calculator.results['stress'] = (
            torch.matmul(calculator._cm.coords.grad.T, calculator._cm.coords) / calculator._cm.box_volume
        ).detach().cpu().numpy() * (Hartree / Bohr**3)
        
        if calculator._cm.box.grad is not None:
            calculator.results['stress'] = calculator.results['stress'] + ((
                torch.matmul(calculator._cm.box.grad.T, calculator._cm.box)
             ) / calculator._cm.box_volume).detach().cpu().numpy() * (Hartree / Bohr**3)
             
        # Clear gradients immediately after use
        calculator._cm.coords.grad = None
        if calculator._cm.box.grad is not None:
            calculator._cm.box.grad = None
        
        # Detach energy tensors
        for key in calculator._energies:
            if isinstance(calculator._energies[key], torch.Tensor):
                calculator._energies[key] = calculator._energies[key].detach()
                
        # Force garbage collection periodically
        if getattr(calculator, '_step_counter', 0) % 5 == 0:
            monitor.force_gc()
            
        # Increment step counter
        calculator._step_counter = getattr(calculator, '_step_counter', 0) + 1
    
    # Replace the method
    calculator._evaluate_ff = evaluate_ff_with_monitoring
    calculator._step_counter = 0
    
    return monitor