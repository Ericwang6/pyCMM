from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple

@dataclass
class NeighborListSettings:
    method: str = "verlet_list"
    cutoff: float = 9.0  # Angstrom
    padding: float = 2.0  # Angstrom
    update_frequency: int = 0
    use_cell_lists: bool = True

@dataclass
class MolecularDynamicsSettings:
    # MD Settings #
    ensemble: str = "NVE"
    timestep: float = 1.0
    n_steps: int = 1000
    driver: str = "ASE"
    
    # Thermostat Settings #
    thermostat: str = "langevin"
    temperature: float = 300.0
    thermostat_coupling: float = 1.0

    # Barostat Settings #
    barostat: str = "none"
    pressure: float = 1.0
    random_seed: Optional[int] = None

    # IO Settings #
    trajectory_frequency: int = 100
    output_frequency: int = 100

@dataclass
class Settings:
    """
    Main settings container for molecular simulations.
    
    Holds various settings objects as a dictionary that can be extended with
    custom settings classes for different simulation types.
    """
    
    # Dictionary of settings objects for different components
    components: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if "neighbor_list" not in self.components:
            self.components["neighbor_list"] = NeighborListSettings()
    
    def get(self, component_name: str) -> Any:
        if component_name not in self.components:
            raise KeyError(f"Settings for '{component_name}' not found")
        return self.components[component_name]
    
    def add(self, component_name: str, settings: Any) -> None:
        self.components[component_name] = settings
    
    def add_neighbor_list(self, settings: NeighborListSettings) -> None:
        self.components["neighbor_list"] = settings