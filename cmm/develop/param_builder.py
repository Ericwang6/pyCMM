import json
from typing import Dict, List, Union, Optional, Any


class ParameterBuilder:
    """A class for constructing JSON files representing force field parameters."""
    
    def __init__(self):
        """Initialize the force field builder with empty parameter dictionaries."""
        self.data = {
            "atomtypes": {},
            "atomic_params": [],
            "pair_params": [],
            "bond_params": [],
            "angle_params": []
        }
    
    def add_molecule(self, molecule_name: str, atom_mapping: Dict[str, str]):
        """
        Add atom type mappings for a molecule.
        
        Args:
            molecule_name: Name of the molecule (e.g., "HOH")
            atom_mapping: Dictionary mapping atom labels to atom types
                         (e.g., {"O": "ow", "H1": "hw", "H2": "hw"})
        """
        self.data["atomtypes"][molecule_name.upper()] = atom_mapping
        return self
    
    def add_atom_type(self, 
                      type_name: str,
                      Z: Optional[float] = None,
                      mono: Optional[float] = None,
                      dipo: Optional[List[float]] = None,
                      quad_s: Optional[List[float]] = None,
                      b_elec: Optional[float] = None,
                      axis_type: Optional[str] = None,
                      z_atom: Optional[str] = None,
                      x_atom: Optional[str] = None,
                      y_atom: Optional[str] = None,
                      b_pauli: Optional[float] = None,
                      q_pauli: Optional[float] = None,
                      Kdipo_pauli: Optional[float] = None,
                      Kquad_pauli: Optional[float] = None,
                      C6_disp: Optional[float] = None,
                      b_disp: Optional[float] = None,
                      alpha: Optional[List[float]] = None,
                      eta: Optional[float] = None,
                      b_xpol: Optional[float] = None,
                      q_xpol: Optional[float] = None,
                      Kdipo_xpol: Optional[float] = None,
                      Kquad_xpol: Optional[float] = None,
                      b_ct: Optional[float] = None,
                      q_ct_acc: Optional[float] = None,
                      Kdipo_ct_acc: Optional[float] = None,
                      Kquad_ct_acc: Optional[float] = None,
                      q_ct_don: Optional[float] = None,
                      Kdipo_ct_don: Optional[float] = None,
                      Kquad_ct_don: Optional[float] = None,
                      **kwargs):
        """
        Add atomic parameters for an atom type.
        
        Args:
            type_name: Atom type identifier (e.g., "ow", "hw")
            Z: Atomic charge parameter
            mono: Monopole parameter
            dipo: Dipole parameters [x, y, z]
            quad_s: Quadrupole parameters (5 values)
            b_elec: Electronic b parameter
            axis_type: Axis type (e.g., "Bisector", "ZThenX")
            z_atom: Z-axis atom reference
            x_atom: X-axis atom reference
            y_atom: Y-axis atom reference (optional)
            ... (all other optional parameters)
            **kwargs: Any additional custom parameters
        """
        params = {
            "type": type_name.lower(),
            "Z": Z,
            "mono": mono,
            "dipo": dipo,
            "quad_s": quad_s,
            "b_elec": b_elec,
            "axis_type": axis_type,
            "z_atom": z_atom,
            "x_atom": x_atom,
            "y_atom": y_atom,
            "b_pauli": b_pauli,
            "q_pauli": q_pauli,
            "Kdipo_pauli": Kdipo_pauli,
            "Kquad_pauli": Kquad_pauli,
            "C6_disp": C6_disp,
            "b_disp": b_disp,
            "alpha": alpha,
            "eta": eta,
            "b_xpol": b_xpol,
            "q_xpol": q_xpol,
            "Kdipo_xpol": Kdipo_xpol,
            "Kquad_xpol": Kquad_xpol,
            "b_ct": b_ct,
            "q_ct_acc": q_ct_acc,
            "Kdipo_ct_acc": Kdipo_ct_acc,
            "Kquad_ct_acc": Kquad_ct_acc,
            "q_ct_don": q_ct_don,
            "Kdipo_ct_don": Kdipo_ct_don,
            "Kquad_ct_don": Kquad_ct_don
        }
        
        # Add any additional custom parameters
        params.update(kwargs)
        
        self.data["atomic_params"].append(params)
        return self
    
    def add_pair_params(self, type_pair: List[str], eps_ct: float, **kwargs):
        """
        Add pair interaction parameters.
        
        Args:
            type_pair: List of two atom types (e.g., ["ow", "hw"])
            eps_ct: Epsilon CT parameter
            **kwargs: Any additional custom parameters
        """
        params = {
            "type": [type_pair[0].lower(), type_pair[1].lower()],
            "eps_ct": eps_ct
        }
        params.update(kwargs)
        self.data["pair_params"].append(params)
        return self
    
    def add_bond_params(self, type_pair: List[str], r_eq: float, 
                       j_cf_pauli: Optional[float] = None,
                       j_cf: Optional[float] = None,
                       **kwargs):
        """
        Add bond parameters.
        
        Args:
            type_pair: List of two atom types forming the bond
            r_eq: Equilibrium bond distance
            j_cf_pauli: J CF Pauli parameter
            j_cf: J CF parameter
            **kwargs: Any additional custom parameters
        """
        params = {
            "type": [type_pair[0].lower(), type_pair[1].lower()],
            "r_eq": r_eq
        }
        
        if j_cf_pauli is not None:
            params["j_cf_pauli"] = j_cf_pauli
        if j_cf is not None:
            params["j_cf"] = j_cf
            
        params.update(kwargs)
        self.data["bond_params"].append(params)
        return self
    
    def add_angle_params(self, type_triplet: List[str], theta_eq: float,
                        j_cf_angle: Optional[float] = None,
                        j_cf_bb: Optional[float] = None,
                        **kwargs):
        """
        Add angle parameters.
        
        Args:
            type_triplet: List of three atom types forming the angle
            theta_eq: Equilibrium angle (in radians)
            j_cf_angle: J CF angle parameter (optional)
            j_cf_bb: J CF bond-bond parameter (optional)
            **kwargs: Any additional custom parameters
        """
        params = {
            "type": [type_triplet[0].lower(), type_triplet[1].lower(), type_triplet[2].lower()],
            "theta_eq": theta_eq
        }
        
        if j_cf_angle is not None:
            params["j_cf_angle"] = j_cf_angle
        if j_cf_bb is not None:
            params["j_cf_bb"] = j_cf_bb
            
        params.update(kwargs)
        self.data["angle_params"].append(params)
        return self
    
    def to_json(self, indent: int = 4) -> str:
        """
        Convert the force field data to a JSON string.
        
        Args:
            indent: Number of spaces for indentation (default: 4)
            
        Returns:
            JSON string representation of the force field
        """
        return json.dumps(self.data, indent=indent)
    
    def save_to_file(self, filename: str, indent: int = 4):
        """
        Save the force field data to a JSON file.
        
        Args:
            filename: Path to the output JSON file
            indent: Number of spaces for indentation (default: 4)
        """
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=indent)
        return self
    
    def load_from_file(self, filename: str):
        """
        Load force field data from a JSON file.
        
        Args:
            filename: Path to the input JSON file
        """
        with open(filename, 'r') as f:
            self.data = json.load(f)
        return self
    
    def get_atom_type(self, type_name: str) -> Optional[Dict[str, Any]]:
        """
        Get parameters for a specific atom type.
        
        Args:
            type_name: Atom type identifier
            
        Returns:
            Dictionary of parameters for the atom type, or None if not found
        """
        for atom_params in self.data["atomic_params"]:
            if atom_params["type"] == type_name:
                return atom_params
        return None
    
    def clear(self):
        """Clear all force field data."""
        self.data = {
            "atomtypes": {},
            "atomic_params": [],
            "pair_params": [],
            "bond_params": [],
            "angle_params": []
        }
        return self
    
    def __repr__(self) -> str:
        """String representation of the builder."""
        return f"ParameterBuilder(atomtypes={len(self.data['atomtypes'])}, " \
               f"atomic_params={len(self.data['atomic_params'])}, " \
               f"pair_params={len(self.data['pair_params'])}, " \
               f"bond_params={len(self.data['bond_params'])}, " \
               f"angle_params={len(self.data['angle_params'])})"


# Example usage
if __name__ == "__main__":
    # Create a builder instance
    builder = ParameterBuilder()
    
    # Add molecule atom types
    builder.add_molecule("HOH", {
        "O": "ow",
        "H1": "hw",
        "H2": "hw"
    })
    builder.add_molecule("NA", {
        "Na": "na+",
    })
    
    # Add oxygen atom type parameters
    builder.add_atom_type(
        type_name="ow",
        Z=5.5473,
        mono=-0.390896,
        dipo=[0.0, 0.0, -0.094298],
        quad_s=[-0.330685, 0.0, 0.0, 0.869923, 0.0],
        b_elec=2.3635,
        axis_type="Bisector",
        z_atom="hw",
        x_atom="hw",
        y_atom="",
        b_pauli=1.9550,
        q_pauli=4.5998,
        Kdipo_pauli=-3.4278,
        Kquad_pauli=-0.9743,
        C6_disp=35.8289,
        b_disp=1.84302,
        alpha=[4.45992, 0.0, 0.0, 6.07259, 0.0, 4.55391],
        eta=0.0,
        b_xpol=2.5447,
        q_xpol=0.8491,
        Kdipo_xpol=1.0026,
        Kquad_xpol=-0.3606,
        b_ct=1.89485,
        q_ct_acc=-0.67857,
        Kdipo_ct_acc=0.0,
        Kquad_ct_acc=0.0,
        q_ct_don=0.757752,
        Kdipo_ct_don=-0.512036,
        Kquad_ct_don=-0.208186
    )
    
    # Add hydrogen atom type parameters
    builder.add_atom_type(
        type_name="hw",
        Z=1.0031,
        mono=0.195448,
        dipo=[0.0910288, 0.0, -0.207851],
        quad_s=[-0.0739388, 0.0929482, 0.0, 0.00532425, 0.0],
        b_elec=2.5154,
        axis_type="ZThenX",
        z_atom="ow",
        x_atom="hw",
        y_atom="",
        b_pauli=2.5732,
        q_pauli=0.9290,
        Kdipo_pauli=-0.6437,
        Kquad_pauli=-0.6274,
        C6_disp=1.98954,
        b_disp=1.30993,
        alpha=[2.22001, 0.0, 0.0, 1.66835, 0.0, 0.183855],
        eta=0.561535,
        b_xpol=3.5033,
        q_xpol=0.7142,
        Kdipo_xpol=0.2058,
        Kquad_xpol=-0.2712,
        b_ct=2.36763,
        q_ct_acc=1.36735,
        Kdipo_ct_acc=0.0,
        Kquad_ct_acc=0.0,
        q_ct_don=0.00888982,
        Kdipo_ct_don=-0.0511668,
        Kquad_ct_don=0.0568152
    )

    builder.add_atom_type(
        type_name="na+",
        Z=3.5489,
        mono=1.0,
        dipo=[0.0, 0.0, 0.0],
        quad_s=[0.0, 0.0, 0.0, 0.0, 0.0],
        b_elec=2.59626,
        axis_type="NoAxisType",
        z_atom="",
        x_atom="",
        y_atom="",
        b_pauli=2.5732,
        q_pauli=0.9290,
        Kdipo_pauli=0.0,
        Kquad_pauli=0.0,
        C6_disp=1.98954,
        b_disp=1.30993,
        alpha=[0.9542199, 0.0, 0.0, 0.9542199, 0.0, 0.9542199],
        eta=0.0,
        b_xpol=2.04028,
        q_xpol=0.200089,
        Kdipo_xpol=0.0,
        Kquad_xpol=0.0,
        b_ct=1.876471,
        q_ct_acc=1.07641,
        Kdipo_ct_acc=0.0,
        Kquad_ct_acc=0.0,
        q_ct_don=0.167905,
        Kdipo_ct_don=0.0,
        Kquad_ct_don=0.0
    )
    
    # Add pair parameters
    builder.add_pair_params(["ow", "hw"], eps_ct=2.624816590940708)
    builder.add_pair_params(["ow", "na+"], eps_ct=1.036375753)
    
    # Add bond parameters
    builder.add_bond_params(
        ["ow", "hw"],
        r_eq=1.81211318,
        j_cf_pauli=0.0911036,
        j_cf=-0.024794
    )
    
    # Add angle parameters
    builder.add_angle_params(
        ["hw", "ow", "hw"],
        theta_eq=1.822532146,
        j_cf_angle=0.0220891,
        j_cf_bb=-0.0332338
    )
    
    # Save to file
    builder.save_to_file("forcefield.json")
    
    # Print the JSON
    print(builder.to_json())