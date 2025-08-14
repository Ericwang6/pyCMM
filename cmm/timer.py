import torch
import time

class Timer:
    def __init__(self, device='cuda'):
        self.device = device
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.end_time = time.perf_counter()
        self.elapsed = (self.end_time - self.start_time) * 1000