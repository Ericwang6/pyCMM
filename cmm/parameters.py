import torch
from torch import nn

class ParameterStorage:
    """
    This class defines an interface for storing and retrieving parameters
    for a force field. There are two types of parameters, those which are
    constant and those which are variable. Variable parameters require the
    force field to define how those parameters should be computed. This
    distinction is needed because the parameter derivatives need to be
    tracked for models which have environment-dependent parameters and
    so that parameters can be easily optimized.
    """

    def __init__(self) -> None:
        self._parameters = {} # Maps a string to a torch.Tensor
    
    def register_parameters(self, name: str, params: torch.Tensor):
        self._parameters[name] = params

    def checkout_parameters(self, name: str):
        return self._parameters[name]

class Variable(nn.Module):
    def __init__(self):
        super(Variable, self).__init__()
        


if __name__ == "__main__":

    #params = ParameterStorage()
    #params.register_parameter("ke", 10)
    #A = params.checkout_parameter("ke")
    #print(A)
    #A[0] = 2.0
    #B = params.checkout_parameter("ke")
    #print(B)
    #print(A is B)
        