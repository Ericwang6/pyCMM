import pytest
import torch
import numpy as np
import os
import tempfile

from cmm.units import HARTREE2EV, BOHR2ANG, BOHR2NM
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM
from cmm.interfaces import CMM_ASE
from cmm.logger import Logger

from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs, bar

def test_md_logger_basic():
    """Test basic functionality of Logger"""
    torch.set_default_dtype(torch.float64)

    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load test system
        coords, atom_types, bonds, labels = read_from_tinker_xyz(
            os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), 
            requires_grad=True
        )
        
        # Setup atom types
        atom_indices_to_names = {0: "O_water", 1: "H_water"}
        atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
        
        # Initialize system
        box = torch.tensor(np.eye(3) * 20.0 / BOHR2ANG, requires_grad=True)
        cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        pairs, _, _ = cm.get_distances_vectors_and_pairs()
        ff = CMM()
        
        # Create parameters
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )
        
        # Create ASE calculator
        calculator = CMM_ASE(ff, cm, topology, parameters, output_folder=temp_dir)
        
        # Define custom property functions
        custom_properties = {
            "max_force": lambda cm, ff, ase_calc: float(np.max(np.abs(ase_calc.results['forces'])))
        }
        
        # Create logger with custom properties
        logger = Logger(
            cm=cm,
            ff=ff,
            ase_calculator=calculator,
            log_interval=1,  # Log every step for testing
            output_folder=temp_dir,
            properties=["step", "time", "temperature", "energy_total", "dipole_magnitude", "max_force"],
            custom_properties=custom_properties
        )
        
        # Test initial log
        values = logger.log(step=0, force=True)
        assert "step" in values
        assert values["step"] == 0
        
        # Initialize velocities for MD
        MaxwellBoltzmannDistribution(calculator.atoms, temperature_K=300, force_temp=True)
        
        # Create dynamics
        dyn = VelocityVerlet(calculator.atoms, 0.5 * fs)
        
        # Attach logger to dynamics
        logger.attach_to_ase_dynamics(dyn)
        
        # Run a few steps
        dyn.run(5)
        
        # Check log file was created
        log_file = os.path.join(temp_dir, "md_log.txt")
        assert os.path.exists(log_file)
        
        # Check log file has correct number of lines (header + steps)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Count non-comment, non-empty lines
            data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
            assert len(data_lines) >= 5  # At least 5 steps logged

        # Test energy component logging
        logger.log_energy_components(detailed=True)
        
        # Test saving state
        state_file = os.path.join(temp_dir, "logger_state.json")
        logger.save_state(filename=state_file)
        assert os.path.exists(state_file)


def test_md_logger_water_box():
    """Test Logger with a larger water box system"""
    torch.set_default_dtype(torch.float64)
    
    # Skip test if it takes too long
    #pytest.skip("Skipping water box test to save time")
    
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        print(temp_dir)

        # Load test system - water box
        coords, atom_types, bonds, labels = read_from_tinker_xyz(
            os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), 
            requires_grad=True
        )
        
        # Setup atom types
        atom_indices_to_names = {0: "O_water", 1: "H_water"}
        atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
        
        # Initialize system
        box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True)
        cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        pairs, _, _ = cm.get_distances_vectors_and_pairs()
        ff = CMM(use_ewald=True)
        
        # Create parameters
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )
        
        # Create ASE calculator
        calculator = CMM_ASE(ff, cm, topology, parameters, output_folder=temp_dir)
        
        # Create logger
        logger = Logger(
            cm=cm,
            ff=ff,
            ase_calculator=calculator,
            log_interval=2,
            output_folder=temp_dir,
            properties=[
                "step", "temperature", "energy_total",
                "kinetic_energy", "volume", "density", "pressure",
                "dipole_magnitude"
            ]
        )
        
        # Initialize velocities for MD
        MaxwellBoltzmannDistribution(calculator.atoms, temperature_K=300, force_temp=True)
        
        # Create dynamics
        dyn = NPTBerendsen(calculator.atoms, 0.5 * fs, temperature_K=300,
                   taut=100 * fs, pressure_au=1.01325 * bar,
                   taup=1000 * fs, compressibility_au=4.57e-5 / bar)
        logger.attach_to_ase_dynamics(dyn)
        dyn.run(6)

        # HERE: Add in logging for dipole moment of the cell.
        # Then submit NVT calculations at a range of temperatures.
        # Copmute dielectric constant.
        # Question: Are the pressure calculations going to be wrong
        # if an atom is not in the primitve cell?
        
        # Check that logging worked
        log_file = os.path.join(temp_dir, "md_log.txt")
        assert os.path.exists(log_file)
        
        # Verify temperature is reasonable
        assert 250 < calculator.atoms.get_temperature() < 350


if __name__ == "__main__":
    # Run with more steps and output when run directly
    test_md_logger_basic()