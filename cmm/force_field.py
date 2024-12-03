import torch
from .multipole import computeCartesianQuadrupoles

# NOTE(JOE): The design of this object is still up in the air. I think that we could
# allow inheritance for the purpose of making it really trivial to set
# up a force field. This just saves the user having to call the appropriate
# set up functions manually I guess? Custom force fields can just work with
# the base class I think. Ultimately all that this object does is hold
# onto a list of functions we need to call and all of the parameters
# needed to pass to the Parameterizer to populate the parameter arrays.
# Also, in the future, we will add the option to specify parameters
# from a file or dictionary or something.

# TODO: The axis types are specified as follows:
# 0 = Identity
# 1 = z-then-x
# 2 = bisector
# That's all we have for now. The axis type should really be specified by the
# force field by using a mapping from the atom type to the axis type. Don't have that yet.

class ForceField:
    def __init__(self) -> None:
        self._params = {} # name of param to torch tensor indexed by atom type
        self._terms = [] # List of functions to evaluate, in order.

class CMM(ForceField):
    def __init__(self) -> None:
        super().__init__()
        self._build()
    
    def _build(self):
        Z = torch.tensor([3.61565, 0.93619])
        mono = torch.tensor([-0.390896, 0.195448])
        qShell = mono - Z
        dipo = torch.tensor([
            [0.0,       0.0, -0.094298],
            [0.0910288, 0.0, -0.207851]
        ])
        quad_s = torch.tensor([
            # Q20,       Q21c,      Q21s, Q22c,       Q22s
            [-0.330685,  0.0,       0.0,  0.869923,   0.0],
            [-0.0739388, 0.0929482, 0.0,  0.00532425, 0.0]
        ])
        self._params = {
            # elec
            "Z": Z,
            "q_shell": qShell,
            "mono": mono,
            "dipo": dipo,
            "quad_s": quad_s,
            "quad": computeCartesianQuadrupoles(quad_s),
            "b": torch.tensor([2.13358, 2.33322]),
            # Pauli repulsion
            "b_pauli": torch.tensor([2.1975, 1.96474]),
            "Kmono_pauli": torch.tensor([6.50923, 0.527804]),
            "Kdipo_pauli": torch.tensor([-5.61925, -0.515584]),
            "Kquad_pauli": torch.tensor([-1.56567, -0.440164]),
            # Dispersion
            "C6_disp": torch.tensor([35.8289, 1.98954]),
            "b_disp": torch.tensor([1.84302, 1.30993]),
            # Polarization
            "alpha": torch.tensor([
                [[4.45992, 0.0, 0.0], [0.0, 6.07259, 0.0], [0.0, 0.0, 4.55391]],
                [[2.22001, 0.0, 0.0], [0.0, 1.66835, 0.0], [0.0, 0.0, 0.183855]]
            ]),
            "eta": torch.tensor([6.18699e-6, 0.561535]),
            # Exchange-polarization
            "b_xpol": torch.tensor([2.73582, 2.04028]),
            "Kmono_xpol": torch.tensor([1.26592, 0.200089]),
            "Kdipo_xpol": torch.zeros((2,)),
            "Kquad_xpol": torch.zeros((2,)),
            # Charge Transfer
            "b_ct": torch.tensor([1.89485, 2.36763]),
            "Kmono_ct_acc": torch.tensor([-0.67857, 1.36735]),
            "Kdipo_ct_acc": torch.tensor([0.0, 0.0]),
            "Kquad_ct_acc": torch.tensor([0.0, 0.0]),
            "Kmono_ct_don": torch.tensor([0.757752, 0.00888982]),
            "Kdipo_ct_don": torch.tensor([-0.512036, -0.0511668]),
            "Kquad_ct_don": torch.tensor([-0.208186, 0.0568152]),
            "eps": torch.tensor([[1e15, 0.380979], [0.380979, 1e15]]),
            "axistypes": torch.tensor([2, 1], dtype=torch.long)
        }