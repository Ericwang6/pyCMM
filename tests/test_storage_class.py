import pytest
import torch
from tensordict import TensorDict

def test_storage_rebuild():
    pytest.skip("This is just a scratch space for now.")
    
    natoms = 5
    max_pairs_per_atom = 8
    td = TensorDict(
        {
            "a": torch.full((max_pairs_per_atom,), -1),
            "b": torch.full((max_pairs_per_atom,), -1)
        },
        batch_size=[max_pairs_per_atom]
    )
    print(td)
    td["a"][0:4] = torch.randint(0, natoms-1, (4,))
    td["b"][0:5] = torch.randint(0, natoms-1, (5,))
    print(td["a"])
    print(td["b"])

    print(td["a"][torch.where(td["a"] != -1, True, False)])
    index_masks = torch.vmap(lambda key, td : torch.where(td[key] != -1, True, False))(td.keys(), td)
    print(index_masks)