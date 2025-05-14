import time
import torch
import threading
import statistics
from collections import defaultdict

# Thread-local storage for nested timing contexts and
# global storage for timing statistics
_local = threading.local()
_timings = defaultdict(list)

class TimingContext:
    def __init__(self, name, sync_cuda=True):
        self.name = name
        self.sync_cuda = sync_cuda
        
    def __enter__(self):
        # Initialize nesting level if not already set
        if not hasattr(_local, 'level'):
            _local.level = 0
        
        # Synchronize CUDA before timing if needed
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        self.start = time.perf_counter()
        _local.level += 1
        return self
        
    def __exit__(self, *args):
        # Synchronize CUDA again to ensure all operations completed
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        elapsed = time.perf_counter() - self.start
        _timings[self.name].append(elapsed)
        _local.level -= 1

def get_timing_stats():
    stats = {}
    for name, times in _timings.items():
        if not times:
            continue
        stats[name] = {
            'count': len(times[2:]) if len(times) > 2 else 0,
            'total': sum(times[2:]) if len(times) > 2 else 0,
            'mean': statistics.mean(times[2:]) if len(times) > 2 else 0,
            'median': statistics.median(times[2:]) if len(times) > 2 else 0,
            'min': min(times[2:]) if len(times) > 2 else 0,
            'max': max(times[2:]) if len(times) > 2 else 0,
            'std': statistics.stdev(times[2:]) if len(times) > 3 else 0
        }
    return stats

def print_timing_stats():
    # TODO: Make it so that we split and only show the current level and previous level in the name.
    # Put some ... to indicate the omitted levels and indent according to number of skipped levels.
    stats = get_timing_stats()
    if not stats:
        print("No timing statistics available.")
        return
        
    print("\n===== Timing Statistics =====")
    print(f"{'Section':<45} | {'Count':>8} | {'Total (s)':>10} | {'Mean (ms)':>10} | {'Median (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10} | {'Std (ms)':>10}")
    print("-" * 135)
    
    for name, data in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"{name:<45} | {data['count']:>8} | {data['total']:>10.4f} | {data['mean']*1000:>10.2f} | {data['median']*1000:>10.2f} | {data['min']*1000:>10.2f} | {data['max']*1000:>10.2f} | {data['std']*1000:>10.2f}")

def reset_timings():
    _timings.clear()