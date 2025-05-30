import torch

# NOTE(JOE): In the future, this will probably use the tensor dict class and we
# will use a maximum size to pre-allocate the memory for parameters, etc.
# There will also be opportunities for splitting up the storage into multiple
# dictionaries of different sizes. Basically, there will be a real API and
# actual optimizations beyond just putting stuff in a dictionary. But for now,
# we are starting simple.

class Storage:
    def __init__(self) -> None:
        self.data = {}

    def add(self, name: str, data: torch.Tensor):
        self.data[name] = data

    def get(self, name: str):
        return self.data[name]