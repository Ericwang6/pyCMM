import numpy as np

# NOTE(JOE): Do not trust these numbers definitively. They come from Claude 4.0 Sonnet on 5/30/25.
# For now they look reasonable enough, but we should obviously fill these out with definitive literature
# values soon. The ATOMIC_NUMBER_TO_SYMBOL table is also AI generated so could have errors.
COVALENT_RADII = {
    'H': 0.37,   'He': 0.32,
    'Li': 1.34,  'Be': 0.90,  'B': 0.82,   'C': 0.77,   'N': 0.75,   'O': 0.73,   'F': 0.71,   'Ne': 0.69,
    'Na': 1.54,  'Mg': 1.30,  'Al': 1.18,  'Si': 1.11,  'P': 1.06,   'S': 1.02,   'Cl': 0.99,  'Ar': 0.97,
    'K': 1.96,   'Ca': 1.74,  'Sc': 1.44,  'Ti': 1.36,  'V': 1.25,   'Cr': 1.27,  'Mn': 1.39,  'Fe': 1.25,
    'Co': 1.26,  'Ni': 1.21,  'Cu': 1.38,  'Zn': 1.31,  'Ga': 1.26,  'Ge': 1.22,  'As': 1.19,  'Se': 1.16,
    'Br': 1.14,  'Kr': 1.10,  'Rb': 2.11,  'Sr': 1.92,  'Y': 1.62,   'Zr': 1.48,  'Nb': 1.37,  'Mo': 1.45,
    'Tc': 1.56,  'Ru': 1.26,  'Rh': 1.35,  'Pd': 1.31,  'Ag': 1.53,  'Cd': 1.48,  'In': 1.44,  'Sn': 1.41,
    'Sb': 1.38,  'Te': 1.35,  'I': 1.33,   'Xe': 1.30
}

ATOMIC_NUMBER_TO_SYMBOL = {
    1: 'H',   2: 'He',  3: 'Li',  4: 'Be',  5: 'B',   6: 'C',   7: 'N',   8: 'O',   9: 'F',   10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',  16: 'S',  17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V',  24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',  40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',  75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
    101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds',
    111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
}

SYMBOL_TO_ATOMIC_NUMBER = {symbol: number for number, symbol in ATOMIC_NUMBER_TO_SYMBOL.items()}

def atomic_number_to_symbol(atomic_number):
    if atomic_number not in ATOMIC_NUMBER_TO_SYMBOL:
        raise ValueError(f"Invalid atomic number: {atomic_number}. Must be between 1 and 118.")
    return ATOMIC_NUMBER_TO_SYMBOL[atomic_number]

def symbol_to_atomic_number(symbol):
    symbol = symbol.capitalize()  # Handle case variations
    if symbol not in SYMBOL_TO_ATOMIC_NUMBER:
        raise ValueError(f"Invalid atomic symbol: {symbol}")
    return SYMBOL_TO_ATOMIC_NUMBER[symbol]

def convert_atomic_numbers_to_labels(atomic_numbers):
    return [atomic_number_to_symbol(num) for num in atomic_numbers]

def guess_bond_connectivity(coordinates, atomic_labels, covalent_radii=COVALENT_RADII, tolerance=0.3):
    n_atoms = len(atomic_labels)
    if coordinates.shape != (n_atoms, 3):
        raise ValueError("Coordinates shape must be (n_atoms, 3)")
    
    ion_labels = ['Li', 'Na', 'K', 'Rb', 'Cs', 'F', 'Cl', 'Br', 'I', 'Mg', 'Ca']

    bonded_pairs = []
    for i in range(n_atoms):
        if atomic_labels[i] in ion_labels:
            continue
        for j in range(i + 1, n_atoms):
            if atomic_labels[j] in ion_labels:
                continue

            radius_i = covalent_radii.get(atomic_labels[i])
            radius_j = covalent_radii.get(atomic_labels[j])
            
            if radius_i is None or radius_j is None:
                print(f"Warning: Covalent radius not found for {atomic_labels[i]} or {atomic_labels[j]}")
                continue

            diff = coordinates[i] - coordinates[j]
            actual_distance = np.sqrt(np.sum(diff**2))

            expected_distance = radius_i + radius_j + tolerance
            if actual_distance <= expected_distance:
                bonded_pairs.append([i, j])
    
    return np.array(bonded_pairs).T