import torch
from typing import List, Tuple
import numpy as np
from .units import BOHR2ANG
from .atom_types import get_expected_connectivities
import tempfile
import os
from contextlib import contextmanager

#def write_xyz(outfile: str, labels: List[str], coords: torch.Tensor) -> None:
#    """
#    Writes a plain xyz file.
#    
#    Args:
#        outfile (str): Name of output file.
#        labels (List[str]): Element labels of all atoms (N,).
#        coords (torch.Tensor): Positions of all atoms (N, 3).
#    """
#
#    natoms = coords.size()[0]
#    assert natoms == len(labels), "Number of atom labels and coordinates do not match."
#    with open(outfile, "w") as f:
#        header = str(natoms) + "\n\n"
#        f.write(header)
#        for i in range(natoms):
#            vec = coords[i, :]
#            line = labels[i] + " " + str(vec[0].item()) + " " + str(vec[1].item()) + " " + str(vec[2].item()) + "\n"
#            f.write(line)

def write_xyz(filename, labels_list, coords_list, comments=None):
    if len(labels_list) != len(coords_list):
        raise ValueError("Number of label lists must match number of coordinate arrays")
    
    with open(filename, 'w') as f:
        for i, (labels, coords) in enumerate(zip(labels_list, coords_list)):
            if len(labels) != coords.shape[0]:
                raise ValueError(f"Frame {i}: Number of labels ({len(labels)}) must match "
                               f"number of coordinates ({coords.shape[0]})")
            
            if coords.shape[1] != 3:
                raise ValueError(f"Frame {i}: Coordinates must have shape (N, 3), "
                               f"got {coords.shape}")
            
            # Write number of atoms
            n_atoms = len(labels)
            f.write(f"{n_atoms}\n")
            
            # Write comment line
            if comments is not None and i < len(comments):
                f.write(f"{comments[i]}\n")
            else:
                f.write(f"Frame {i + 1}\n")
            
            # Write atomic coordinates
            for label, coord in zip(labels, coords):
                f.write(f"{label:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

def read_xyz(filepath: str) -> Tuple[List[np.ndarray], List[List[str]]]:
    """
    Read XYZ formatted file containing molecular structures.
    
    Parameters:
    -----------
    filepath : str
        Path to the XYZ file
        
    Returns:
    --------
    coordinates : List[np.ndarray]
        List of numpy arrays, each with shape (n_atoms, 3) containing atomic coordinates
    labels : List[List[str]]
        List of lists, each containing atom labels for a frame
    """
    coordinates = []
    labels = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
            
        # Read number of atoms
        try:
            n_atoms = int(lines[i].strip())
        except ValueError:
            raise ValueError(f"Expected number of atoms at line {i+1}, got: {lines[i].strip()}")
        
        # Skip comment line
        i += 1
        if i >= len(lines):
            raise ValueError("Unexpected end of file after atom count")
        
        # Read atoms for this frame
        frame_coords = []
        frame_labels = []
        
        for j in range(n_atoms):
            i += 1
            if i >= len(lines):
                raise ValueError(f"Unexpected end of file while reading frame")
            
            # Parse atom line
            parts = lines[i].strip().split()
            if len(parts) < 4:
                raise ValueError(f"Invalid atom line at {i+1}: {lines[i].strip()}")
            
            # Extract label and coordinates
            label = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                raise ValueError(f"Invalid coordinates at line {i+1}: {lines[i].strip()}")
            
            frame_labels.append(label)
            frame_coords.append([x, y, z])
        
        # Store frame data
        coordinates.append(np.array(frame_coords))
        labels.append(frame_labels)
        
        i += 1
    
    return coordinates, labels


def read_xyz_tinker(infile: str):
    """
    Reads an xyz formatted file following the tinker convention.
    Specifically, the first column is the atom number, second is
    the atom labels, then xyz coordinates, then the integer atom type,
    followed by the atom numbers to which this atom is connected.
    An atom need not be connected to any other atoms.
    """
    atom_numbers = []
    atom_labels = []
    coords = []
    atom_types = []

    with open(infile, "r") as f:
        lines = f.readlines()
        header = lines.pop(0)
        
        # The point here is that the connectivity should be symmetric
        # when transposed. We only actually need the bonds in one
        # direction (i<j by choice), but we keep track of both when parsing 
        # to make sure that the topology is well-defined.
        natoms = int(header.split()[0])
        bonds_start_i_less_than_j = []
        bonds_end_i_less_than_j = []
        bonds_start_i_greater_than_j = []
        bonds_end_i_greater_than_j = []
        for line in lines:
            split_line = line.split()
            atom_number = int(split_line[0])
            atom_numbers.append(atom_number)
            atom_labels.append(str(split_line[1]))
            coords.append(np.array([split_line[2], split_line[3], split_line[4]], dtype=np.float64))
            atom_types.append(int(split_line[5]))
            if len(split_line) > 6: # The rest of the line is the bonding info
                bonds = split_line[6:]
                for bond_end in bonds:
                    bond_end = int(bond_end)
                    if bond_end > atom_number:
                        bonds_start_i_less_than_j.append(atom_number)
                        bonds_end_i_less_than_j.append(bond_end)
                    elif bond_end < atom_number:
                        bonds_start_i_greater_than_j.append(atom_number)
                        bonds_end_i_greater_than_j.append(bond_end)
                    else:
                        raise ValueError("Tinker formatted xyz file indicates an atom is bonded to itself. Please ensure the input is correct.")
    
    # TODO: Could add option to ignore this check since one could realistically omit connectivity
    # info from the file for atoms which are connected to atoms of a smaller index.
    if bonds_start_i_less_than_j != bonds_end_i_greater_than_j or bonds_start_i_greater_than_j != bonds_end_i_less_than_j:
        raise ValueError("Bond connectivity in tinker xyz file is not symmetric. Please ensure input is correct.")
    
    # Subtract 1 from bond arrays because tinker xyz specifies the first atom starting from 1.
    return atom_labels, np.array(atom_types, dtype=np.int64), np.vstack(coords), np.array([np.array(bonds_start_i_less_than_j) - 1, np.array(bonds_end_i_less_than_j) - 1])

def read_from_tinker_xyz(xyz_file: str, requires_grad=True, device="cpu"):
    labels, atom_types, coords, bonds = read_xyz_tinker(xyz_file)
    permutation = np.argsort(bonds[0], kind='stable') # Make sure sort is stable so equivalent indices don't get swapped.
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    atom_types = torch.tensor(atom_types, dtype=torch.long, requires_grad=False, device=device) - 1
    coords = torch.tensor(coords / BOHR2ANG, dtype=torch.get_default_dtype(), requires_grad=requires_grad, device=device)
    return coords, atom_types, bonds, labels

def extract_frame_as_pdb_string(pdb_file, frame_index):
    """
    Extract a specific frame from a multi-frame PDB file as a string.
    
    Args:
        pdb_file (str): Path to the PDB file
        frame_index (int): Zero-based index of the frame to extract
    
    Returns:
        str: PDB content for the specified frame
    """
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    frame_lines = []
    current_frame = 0
    in_target_frame = False
    has_model_records = any(line.startswith('MODEL') for line in lines)
    
    # Handle files with MODEL/ENDMDL records (multi-frame)
    if has_model_records:
        for line in lines:
            if line.startswith('MODEL'):
                model_num = int(line.split()[1]) if len(line.split()) > 1 else current_frame + 1
                in_target_frame = (current_frame == frame_index)
                if in_target_frame:
                    frame_lines.append(line)
            elif line.startswith('ENDMDL'):
                if in_target_frame:
                    frame_lines.append(line)
                    break
                current_frame += 1
            elif in_target_frame:
                frame_lines.append(line)
            elif not line.startswith(('ATOM', 'HETATM', 'CONECT', 'TER')):
                # Include header lines (HEADER, TITLE, etc.) for all frames
                if current_frame == 0:
                    frame_lines.append(line)
    
    # Handle files without MODEL records (single frame or concatenated frames)
    else:
        # This is trickier - we need to identify frame boundaries
        # Assume frames are separated by END records or coordinate blocks
        atom_blocks = []
        current_block = []
        header_lines = []
        
        for line in lines:
            if line.startswith(('HEADER', 'TITLE', 'COMPND', 'SOURCE', 'REMARK')):
                if not current_block:  # Only collect headers before first atom block
                    header_lines.append(line)
            elif line.startswith(('ATOM', 'HETATM')):
                current_block.append(line)
            elif line.startswith('END') and current_block:
                atom_blocks.append(current_block)
                current_block = []
            elif line.startswith(('CONECT', 'TER')) and current_block:
                current_block.append(line)
        
        # Handle case where file doesn't end with END
        if current_block:
            atom_blocks.append(current_block)
        
        if frame_index < len(atom_blocks):
            frame_lines = header_lines + atom_blocks[frame_index] + ['END\n']
        else:
            raise IndexError(f"Frame {frame_index} not found. File has {len(atom_blocks)} frames.")
    
    if not frame_lines:
        raise IndexError(f"Frame {frame_index} not found in PDB file")
    
    return ''.join(frame_lines)

@contextmanager
def temporary_pdb_file(pdb_content):
    """
    Context manager for creating and cleaning up temporary PDB files.
    
    Args:
        pdb_content (str): PDB file content as string
    
    Yields:
        str: Path to the temporary PDB file
    """
    temp_fd = None
    temp_path = None
    
    try:
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdb', text=True)
        
        # Write content to temporary file
        with os.fdopen(temp_fd, 'w') as temp_file:
            temp_file.write(pdb_content)
            temp_fd = None  # Prevent double-close
        
        yield temp_path
        
    finally:
        # Clean up
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except:
                pass
        
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

def get_masses(labels: List[str]):
    mass_table = {
        'H': 1.00782503223,
        'D': 2.0141017778,
        'C': 11.9999999958,
        'N': 14.003074,
        'O': 15.99491561957,
        'Li': 6.96,
        'Na': 22.98976928,
        'K': 39.0983,
        'Rb': 85.4678,
        'Cs': 132.90545196,
        'Be': 9.0121831,
        'Mg': 24.3055,
        'Ca': 40.078,
        'F': 18.998403163,
        'Cl': 35.450,
        'Br': 79.904,
        'I': 126.90447
    }
    masses = torch.zeros((len(labels),), dtype=torch.float64)
    for i in range(len(labels)):
        masses[i] = mass_table[labels[i]]
    return masses