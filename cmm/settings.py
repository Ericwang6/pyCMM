from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
from .units import BOHR2ANG

@dataclass
class ShortRangeSettings:
    use_switching: bool = True
    cutoff: float = 5.0 / BOHR2ANG
    switching_start_before_cutoff: float = 2.0 # Bohr

@dataclass
class LongRangeElectrostaticsSettings:
    use_long_range: bool = True
    method: str = "ewald"
    cutoff: float = 9.0 / BOHR2ANG
    max_rank: int = 2
    tolerance: float = 1e-6
    alpha: float = 0.0
    k_max: int = 0
    # ^^^ This will be split into nx, ny, nz eventually
    # and be interpreted as the k vector integers or the
    # number of grid points for PME and Ewald respectively.
    use_switching: bool = False
    switching_start_before_cutoff: float = 2.0 # Bohr

@dataclass
class LongRangeDispersionSettings:
    use_long_range: bool = True
    method: str = "lrc"
    use_switching: bool = True
    switching_start_before_cutoff: float = 2.0

@dataclass
class NeighborListSettings:
    method: str = "verlet"
    cutoff: float = 9.0 / BOHR2ANG
    padding: float = 2.0 / BOHR2ANG
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
        if "long_range_electrostatics" not in self.components:
            self.components["long_range_electrostatics"] = LongRangeElectrostaticsSettings()
        if "long_range_dispersion" not in self.components:
            self.components["long_range_dispersion"] = LongRangeDispersionSettings()
    
    def get_long_range_electrostatics_settings(self):
        return self.get("long_range_electrostatics")
    
    def get_long_range_dispersion_settings(self):
        return self.get("long_range_dispersion")
    
    def get_neighbor_list_settings(self):
        return self.get("neighbor_list")

    def get(self, component_name: str) -> Any:
        if component_name not in self.components:
            raise KeyError(f"Settings for '{component_name}' not found")
        return self.components[component_name]
    
    def add(self, component_name: str, settings: Any) -> None:
        self.components[component_name] = settings

    def add_long_range_electrostatics_settings(self,
        use_long_range: bool = True,
        method: str = "ewald",
        cutoff: float = 8.0,
        max_rank: int = 2,
        tolerance: float = 1e-6,
    ) -> None:
        self.components["long_range_electrostatics"] = LongRangeElectrostaticsSettings(
            use_long_range, method, cutoff / BOHR2ANG, max_rank, tolerance
        )

    def add_long_range_dispersion_settings(self,
        use_long_range: bool = True,
        method: str = "lrc",
        use_switching: bool = True
    ) -> None:
        # This shares the cutoff used by the neighbor list #
        self.components["long_range_dispersion"] = LongRangeDispersionSettings(
            use_long_range, method, use_switching
        )
    
    def add_neighbor_list_settings(self,
        method: str = "verlet",
        cutoff: float = 9.0,
        padding: float = 2.0,
        update_frequency: int = 0,
        use_cell_lists: bool = True
    ) -> None:
        self.components["neighbor_list"] = NeighborListSettings(
            method, cutoff / BOHR2ANG, padding / BOHR2ANG,
            update_frequency, use_cell_lists
        )