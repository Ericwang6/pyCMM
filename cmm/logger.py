import os
import time
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Callable, Set
from datetime import datetime
from ase.units import bar

from .coordinate_manager import CoordinateManager
from .force_field import CMM, ForceField
from .units import BOHR2ANG


class Logger:
    """
    Logger for simulations using the CMM force field.
    
    This logger can track and record various properties throughout a simulation,
    including energies, temperatures, pressures, and custom properties.
    It works with both direct CMM calculations and ASE-driven simulations.
    """
    
    def __init__(
        self,
        cm: CoordinateManager,
        ff: ForceField,
        ase_calculator=None,
        log_interval: int = 10,
        output_folder: str = ".",
        log_file: str = "md_log.txt",
        properties: List[str] = None,
        custom_properties: Dict[str, Callable] = None,
    ):
        """
        Initialize the Logger.
        
        Args:
            cm: CoordinateManager instance
            ff: ForceField instance (e.g., CMM)
            ase_calculator: Optional CMM_ASE calculator for MD-specific properties
            log_interval: How often to log (in MD steps)
            output_folder: Directory for log files
            log_file: Name of the main log file
            properties: List of built-in properties to log
            custom_properties: Dictionary of name -> function(cm, ff, ase_calc) to compute custom properties
        """
        self.cm = cm
        self.ff = ff
        self.ase_calculator = ase_calculator
        self.log_interval = log_interval
        self.output_folder = output_folder
        self.log_file_path = os.path.join(output_folder, log_file)
        self.last_step = -1
        self.start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Default properties to log if none specified
        if properties is None:
            self.properties = ["step", "time", "energy_total", "temperature"]
            if ase_calculator is not None:
                self.properties.extend(["kinetic_energy", "potential_energy"])
        else:
            self.properties = properties
        
        # Custom property functions
        self.custom_properties = custom_properties or {}
        
        # Initialize CSV header
        self._initialize_log_file()
        
        # Keep track of available properties
        self._available_properties = self._get_available_properties()
        
        # Warn about unavailable requested properties
        self._check_property_availability()
    
    def _initialize_log_file(self):
        """Initialize the log file with headers."""
        if os.path.exists(self.log_file_path):
            # If the file already exists, we are restarting a simulation
            # and the header should already be in the log file.
            return
        with open(self.log_file_path, 'w') as f:
            # Write header
            header = "# CMM MD Simulation Log\n"
            header += f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            if self.ase_calculator is not None and hasattr(self.ase_calculator, 'atoms'):
                header += f"# System: {len(self.ase_calculator.atoms)} atoms\n"
            else:
                header += f"# System: {self.cm.coords.shape[0]} atoms\n"
            header += "#\n"
            
            # Create CSV header
            csv_header = ",".join(self.properties)
            header += f"# {csv_header}\n"
            
            f.write(header)
    
    def _get_available_properties(self) -> Set[str]:
        """Get set of all available properties that can be logged."""
        available = {
            # Basic properties
            "step", "time", "wall_time",
            
            # Energy components from CMM
            "energy_total", "energy_perm_elec", "energy_pol", "energy_ct_direct",
            "energy_xpol", "energy_pauli", "energy_disp", "energy_deformation",
            "energy_bond", "energy_angle", "energy_bond_bond", "energy_bond_angle",
            "energy_ewald",

            # Properties from CMM
            "dipole_moment", "dipole_magnitude",
            
            # System properties
            "volume", "density", "box_lengths",
            
            # Custom properties (user-defined)
            *self.custom_properties.keys()
        }
        
        # Add ASE-specific properties if available
        if self.ase_calculator is not None:
            available.update({
                "temperature", "pressure", "kinetic_energy", "potential_energy",
                "conserved_energy", "momentum"
            })
        
        return available
    
    def _check_property_availability(self):
        """Check if all requested properties are available and warn if not."""
        for prop in self.properties:
            if prop not in self._available_properties:
                print(f"Warning: Requested property '{prop}' is not available for logging")
    
    def _get_property_value(self, prop: str) -> Any:
        """Get the value of a specific property."""
        # Basic properties
        if prop == "step":
            return self.last_step
        
        #elif prop == "time":
        #    return self.last_step
        
        elif prop == "wall_time":
            return time.time() - self.start_time
        
        # ASE-specific properties
        elif prop == "temperature" and self.ase_calculator is not None:
            return self.ase_calculator.atoms.get_temperature()
        
        elif prop == "kinetic_energy" and self.ase_calculator is not None:
            return self.ase_calculator.atoms.get_kinetic_energy()
        
        elif prop == "potential_energy" and self.ase_calculator is not None:
            return self.ase_calculator.atoms.get_potential_energy()
        
        elif prop == "dipole_moment":
            return self.ase_calculator.get_dipole_moment()
        
        elif prop == "dipole_magnitude":
            return np.linalg.norm(self.ase_calculator.get_dipole_moment())
        
        elif prop == "momentum" and self.ase_calculator is not None:
            return np.linalg.norm(self.ase_calculator.atoms.get_momenta().sum(axis=0))
        
        elif prop == "pressure" and self.ase_calculator is not None:
            if hasattr(self.ase_calculator, 'results') and 'stress' in self.ase_calculator.results:
                stress = self.ase_calculator.results['stress'] + self.ase_calculator.atoms.get_kinetic_stress(voigt=False)
                pressure = -(stress[0, 0] + stress[1, 1] + stress[2, 2]) / 3  # Negative trace of stress tensor
                return pressure / bar / 1.01325
            return None
        
        elif prop == "density" and self.ase_calculator is not None:
            # Calculate actual density using atom masses from ASE
            if self.ase_calculator is not None and hasattr(self.ase_calculator, 'atoms'):
                # Get masses in amu and sum them
                masses = self.ase_calculator.atoms.get_masses()
                total_mass_amu = np.sum(masses)
                # Convert from amu to g
                total_mass_g = total_mass_amu * 1.66053886e-24
                # Get volume in cm^3 (bohr^3 to cm^3)
                volume_cm3 = float(self.cm.box_volume.detach().cpu().numpy()) * (BOHR2ANG * 1e-8)**3
                # Return density in g/cm^3
                return total_mass_g / volume_cm3
        
        # System properties
        elif prop == "volume":
            return float(self.cm.box_volume.detach().cpu().numpy())
        
        elif prop == "box_lengths":
            return self.cm.box_lengths.detach().cpu().numpy().tolist()
        
        # Energy components - extract from the latest calculation
        elif prop.startswith("energy_"):
            energy_type = prop[7:]  # Remove "energy_" prefix
            if hasattr(self.ase_calculator, '_energies') and energy_type in self.ase_calculator._energies:
                return float(self.ase_calculator._energies[energy_type].detach().cpu().numpy())
            return None
        
        # Custom properties
        elif prop in self.custom_properties:
            return self.custom_properties[prop](self.cm, self.ff, self.ase_calculator)
        
        else:
            return None
    
    def log(self, step: Optional[int] = None, force: bool = False, print_to_console: bool = False) -> Dict[str, Any]:
        """
        Log the current state of the simulation.
        
        Args:
            step: Current step number (optional, auto-detected if using ASE)
            force: Whether to force logging even if log_interval hasn't been reached
            
        Returns:
            Dictionary of property values that were logged
        """
        if step is not None:
            self.last_step = step
        else:
            self.last_step += 1
        
        # Check if we should log this step
        if not force and self.last_step % self.log_interval != 0:
            return {}
        
        # Collect property values
        values = {}
        for prop in self.properties:
            values[prop] = self._get_property_value(prop)
        
        # Write to log file
        with open(self.log_file_path, 'a') as f:
            csv_line = ",".join("{:.7f}".format(values.get(prop, "")) for prop in self.properties)
            f.write(f"{csv_line}\n")
        
        # Print to console
        if print_to_console:
            self._print_log(values)
        
        return values
    
    def _print_log(self, values: Dict[str, Any]):
        """Print formatted log information to console."""
        log_parts = []
        
        # Print common properties in a nice format
        if "step" in values:
            log_parts.append(f"Step: {values['step']}")
        
        if "time" in values:
            log_parts.append(f"Time: {values['time']:.2f} fs")
        
        if "temperature" in values and values["temperature"] is not None:
            log_parts.append(f"T: {values['temperature']:.1f} K")
        
        if "energy_total" in values and values["energy_total"] is not None:
            log_parts.append(f"E_tot: {values['energy_total']:.6f}")
        
        if "potential_energy" in values and values["potential_energy"] is not None:
            log_parts.append(f"E_pot: {values['potential_energy']:.6f}")
        
        if "kinetic_energy" in values and values["kinetic_energy"] is not None:
            log_parts.append(f"E_kin: {values['kinetic_energy']:.6f}")
        
        print(" | ".join(log_parts))
    
    def attach_to_ase_dynamics(self, dynamics):
        """
        Attach this logger to an ASE dynamics object.
        
        Args:
            dynamics: ASE dynamics object (e.g., VelocityVerlet)
        """
        def log_wrapper():
            self.log()
        
        # This interval means ASE will check if we want to log at every step
        # but logging will happen at the interval given to the Logger.
        dynamics.attach(log_wrapper, interval=1)
    
    def log_energy_components(self, detailed: bool = False):
        """
        Log a detailed breakdown of energy components.
        
        Args:
            detailed: Whether to log even more detailed components
        """
        if not hasattr(self.ase_calculator, '_energies'):
            print("No energy components available")
            return
        
        energies = self.ase_calculator._energies
        print("\nEnergy Components:")
        print("-----------------")
        
        # Always print main components
        for component in ['total', 'perm_elec', 'pol', 'pauli', 'disp', 'ct_direct', 'deformation']:
            if component in energies:
                value = float(energies[component].detach().cpu())
                print(f"{component.ljust(12)}: {value:.8f}")
        
        # Print additional components if detailed
        if detailed:
            print("\nDetailed Components:")
            print("-------------------")
            for component in ['bond', 'angle', 'bond_bond', 'bond_angle', 'ewald']:
                if component in energies:
                    value = float(energies[component].detach().cpu())
                    print(f"{component.ljust(12)}: {value:.8f}")
    
    def save_state(self, filename: str = None):
        """
        Save the current state of the logger.
        
        Args:
            filename: Name of file to save state to (defaults to md_logger_state.json)
        """
        if filename is None:
            filename = os.path.join(self.output_folder, "md_logger_state.json")
        
        state = {
            "last_step": self.last_step,
            "start_time": self.start_time,
            "properties": self.properties,
            "log_interval": self.log_interval,
            "log_file": os.path.basename(self.log_file_path),
            "output_folder": self.output_folder
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, filename: str, cm: CoordinateManager, ff: ForceField, ase_calculator=None):
        """
        Load a logger from a saved state.
        
        Args:
            filename: Name of file to load state from
            cm: CoordinateManager instance
            ff: ForceField instance
            ase_calculator: Optional CMM_ASE calculator
            
        Returns:
            Logger instance with restored state
        """
        with open(filename, 'r') as f:
            state = json.dump(f)
        
        logger = cls(
            cm=cm,
            ff=ff,
            ase_calculator=ase_calculator,
            log_interval=state["log_interval"],
            output_folder=state["output_folder"],
            log_file=state["log_file"],
            properties=state["properties"]
        )
        
        logger.last_step = state["last_step"] 
        logger.start_time = state["start_time"]
        
        return logger