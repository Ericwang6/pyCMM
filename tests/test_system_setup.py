import torch
import os

from cmm.system import System, create_system_from_ext_xyz_file
from cmm.settings import Settings, MolecularDynamicsSettings, NeighborListSettings

def test_system_setup_from_file():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    md_settings = MolecularDynamicsSettings(
        ensemble="NVT", timestep=0.5,
        n_steps=50, temperature=298.15
    )
    nl_settings = NeighborListSettings(
        padding=1.5
    )
    settings = Settings()
    settings.add_neighbor_list(nl_settings)
    settings.add("md", md_settings)

    create_system_from_ext_xyz_file(os.path.join(os.path.dirname(__file__), "data/water_clusters.xyz"), requires_grad=True, device=device)
    # HERE: Keep working on the system and settings stuff.
    # 1) Load system from extxyz file
    # 2) Make system have same functionality as coordinate manager