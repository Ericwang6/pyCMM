import json, sys, os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def json_to_forcefield_xml(json_data):
    """
    Convert force field JSON data to XML format.
    
    Args:
        json_data: Either a JSON string or a Python dictionary containing the force field data
        
    Returns:
        A formatted XML string
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Create root element
    root = ET.Element("ForceField")
    
    # Process Residues
    if "atomtypes" in data:
        residues = ET.SubElement(root, "Residues")
        for residue_name, atoms in data["atomtypes"].items():
            residue = ET.SubElement(residues, "Residue", name=residue_name)
            for atom_name, atom_type in atoms.items():
                ET.SubElement(residue, "Atom", name=atom_name, type=atom_type)
    
    # Process Bonds
    # NOTE(JOE): Some of these defaults should really be zero but I have
    # hard-coded them to the values for water since those params do not
    # appear in the water file. This script is likely to become obsolete
    # once we migrate to always using the XML format for the fitting.
    # In case it doesn't, we should update the handling of defaults and 
    # look up the equilibrium distances for the bond-bond charge flux from
    # the bond params we have already loaded.
    if "bond_params" in data:
        bonds = ET.SubElement(root, "Bonds")
        for bond in data["bond_params"]:
            bond_attribs = {
                "type1": bond["type"][0],
                "type2": bond["type"][1],
                "r_eq": str(bond.get("r_eq", "")),
                "D": str(bond.get("D", "0.19968199")),
                "k_b": str(bond.get("D", "0.54375456")),
                "j_cf": str(bond.get("j_cf", "0.0")),
                "j_cf_pauli": str(bond.get("j_cf_pauli", "0.0")),
                "k_hardness_b":str(bond.get("k_hardness_b", "0.0")),
                "dip_deriv_1": str(bond.get("dip_deriv_1", "0.16542209")),
                "dip_deriv_2": str(bond.get("dip_deriv_2", "-0.0124584")),
                "ct_slope_1": str(bond.get("ct_slope_1", "0.0")),
                "ct_slope_2": str(bond.get("ct_slope_2", "0.0")),
            }
            ET.SubElement(bonds, "Bond", **bond_attribs)
    
    # Process Angles
    if "angle_params" in data:
        angles = ET.SubElement(root, "Angles")
        for angle in data["angle_params"]:
            angle_attribs = {
                "type1": angle["type"][0],
                "type2": angle["type"][1],
                "type3": angle["type"][2],
                "theta_eq": str(angle.get("theta_eq", "1.822532146")),
                "r_eq_1": str(angle.get("r_eq_1", "1.81211318")),
                "r_eq_2": str(angle.get("r_eq_2", "1.81211318")),
                "k_theta": str(angle.get("k_theta", "0.1722274")),
                "k_bb": str(angle.get("k_bb", "-0.006521268391698836")),
                "k_ba_1": str(angle.get("k_ba_1", "-0.03222549577618679")),
                "k_ba_2": str(angle.get("k_ba_2", "-0.03222549577618679")),
                "j_cf_angle": str(angle.get("j_cf_angle", "0.0220891")),
                "k_hardness_angle": str(angle.get("k_hardness_angle", "0.0")),
                "j_cf_bb": str(angle.get("j_cf_bb", "-0.0332338")),
                "k_hardness_bb": str(angle.get("k_hardness_bb", "0.0")),
            }
            ET.SubElement(angles, "Angle", **angle_attribs)
    
    # Process atomic parameters
    if "atomic_params" in data:
        # Create dictionaries to store different parameter types
        multipoles_data = []
        cp_data = []
        pauli_data = []
        xpol_data = []
        disp_data = []
        ct_direct_data = []
        pol_data = []
        
        for atom in data["atomic_params"]:
            atom_type = atom["type"]
            
            # Multipoles
            multipoles_data.append({
                "type": atom_type,
                "kz": atom.get("z_atom", ""),
                "kx": atom.get("x_atom", ""),
                "ky": atom.get("y_atom", ""),
                "axistype": atom.get("axis_type", ""),
                "c0": str(atom.get("mono", "")),
                "dx": str(atom["dipo"][0]) if "dipo" in atom else "0.0",
                "dy": str(atom["dipo"][1]) if "dipo" in atom else "0.0",
                "dz": str(atom["dipo"][2]) if "dipo" in atom else "0.0",
                "q20": str(atom["quad_s"][0]) if "quad_s" in atom else "0.0",
                "q21c": str(atom["quad_s"][1]) if "quad_s" in atom else "0.0",
                "q21s": str(atom["quad_s"][2]) if "quad_s" in atom else "0.0",
                "q22c": str(atom["quad_s"][3]) if "quad_s" in atom else "0.0",
                "q22s": str(atom["quad_s"][4]) if "quad_s" in atom else "0.0"
            })
            
            # Charge Penetration
            cp_data.append({
                "type": atom_type,
                "Z": str(atom.get("Z", "")),
                "b_elec": str(atom.get("b_elec", ""))
            })
            
            # Pauli Repulsion
            pauli_data.append({
                "type": atom_type,
                "q_pauli": str(atom.get("q_pauli", "0.0")),
                "Kdipo_pauli": str(atom.get("Kdipo_pauli", "0.0")),
                "Kquad_pauli": str(atom.get("Kquad_pauli", "0.0")),
                "b_pauli": str(atom.get("b_pauli", "0.0"))
            })
            
            # Exchange Polarization
            xpol_data.append({
                "type": atom_type,
                "q_xpol": str(atom.get("q_xpol", "0.0")),
                "Kdipo_xpol": str(atom.get("Kdipo_xpol", "0.0")),
                "Kquad_xpol": str(atom.get("Kquad_xpol", "0.0")),
                "b_xpol": str(atom.get("b_xpol", "0.0"))
            })
            
            # Dispersion
            disp_data.append({
                "type": atom_type,
                "C6_disp": str(atom.get("C6_disp", "0.0")),
                "b_disp": str(atom.get("b_disp", "0.0"))
            })
            
            # Charge Transfer Direct
            ct_direct_data.append({
                "type": atom_type,
                "q_ct_acc": str(atom.get("q_ct_acc", "0.0")),
                "q_ct_don": str(atom.get("q_ct_don", "0.0")),
                "Kdipo_ct_acc": str(atom.get("Kdipo_ct_acc", "0.0")),
                "Kdipo_ct_don": str(atom.get("Kdipo_ct_don", "0.0")),
                "Kquad_ct_acc": str(atom.get("Kquad_ct_acc", "0.0")),
                "Kquad_ct_don": str(atom.get("Kquad_ct_don", "0.0")),
                "b_ct": str(atom.get("b_ct", "0.0"))
            })
            
            # Polarization
            if "alpha" in atom:
                pol_data.append({
                    "type": atom_type,
                    "eta": str(atom.get("eta", "")),
                    "alpha_xx": str(atom["alpha"][0]),
                    "alpha_yy": str(atom["alpha"][3]),
                    "alpha_zz": str(atom["alpha"][5]),
                    "alpha_damp_exponent": str(atom.get("alpha_damp_exponent", "0.0")),
                    "alpha_damp_max": str(atom.get("alpha_damp_max", "0.0")),
                })
        
        # Add Multipoles section
        multipoles = ET.SubElement(root, "Multipoles")
        for mp in multipoles_data:
            ET.SubElement(multipoles, "Multipole", **mp)
        
        # Add ChargePenetration section
        cp = ET.SubElement(root, "ChargePenetration")
        for cp_item in cp_data:
            ET.SubElement(cp, "CP", **cp_item)
        
        # Add PauliRepulsion section
        pauli = ET.SubElement(root, "PauliRepulsion")
        for pauli_item in pauli_data:
            ET.SubElement(pauli, "Pauli", **pauli_item)
        
        # Add ExchangePolarization section
        xpol = ET.SubElement(root, "ExchangePolarization")
        for xpol_item in xpol_data:
            ET.SubElement(xpol, "Xpol", **xpol_item)
        
        # Add Dispersion section
        disp = ET.SubElement(root, "Dispersion")
        for disp_item in disp_data:
            ET.SubElement(disp, "Disp", **disp_item)
        
        # Add ChargeTransfer section
        ct = ET.SubElement(root, "ChargeTransfer")
        for ct_item in ct_direct_data:
            ET.SubElement(ct, "Direct", **ct_item)
        
        # Add Indirect charge transfer from pair_params
        if "pair_params" in data:
            for pair in data["pair_params"]:
                indirect_attribs = {
                    "type1": pair["type"][0],
                    "type2": pair["type"][1],
                    "eps_ct": str(pair.get("eps_ct", ""))
                }
                ET.SubElement(ct, "Indirect", **indirect_attribs)
        
        # Add Polarization section
        pol = ET.SubElement(root, "Polarization")
        for pol_item in pol_data:
            ET.SubElement(pol, "Pol", **pol_item)
    
    # Convert to string with pretty formatting
    xml_str = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="    ")
    
    # Remove extra blank lines
    lines = pretty_xml.split('\n')
    lines = [line for line in lines if line.strip()]
    # Skip the XML declaration line
    return '\n'.join(lines[1:])


# Example usage
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        basename = os.path.basename(filepath)
        outname, _ = os.path.splitext(basename)
        outname = outname + ".xml"
    else:
        print("Please provide a JSON parameter file. Exiting.")
        sys,exit(0)
    with open(filepath, 'r') as f:
        json_data = json.load(f)
    
    # Convert to XML
    xml_output = json_to_forcefield_xml(json_data)
    print(xml_output)
    
    # Save to file
    with open(outname, "w") as f:
        f.write(xml_output)