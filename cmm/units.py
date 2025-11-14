from scipy import constants

# Using 0.529177 for BOHR2ANG and 627.51 for HARTREE2KCAL makes the comparison with Julia CMM exact.
BOHR2ANG = constants.value("atomic unit of length") * 1e10
BOHR2NM = constants.value("atomic unit of length") * 1e9

ELE_CHG = constants.elementary_charge
AVOGADRO = constants.Avogadro

INV_4PI_EPS0 = 8.987551e9 * ELE_CHG * ELE_CHG * 1e7 * AVOGADRO / 4.184 # in kcal/mol * A / e^-2

DEBYE2EA = 0.2081943
DEBYE2EBOHR = DEBYE2EA / BOHR2ANG
DEBYE2AU = DEBYE2EBOHR

HARTREE2KJ = constants.value("atomic unit of energy") * AVOGADRO / 1000
HARTREE2KCAL = HARTREE2KJ / 4.184
HARTREE2EV = constants.value("Hartree energy in eV")
EV2KCAL = HARTREE2KCAL / HARTREE2EV

EPSILON0 = constants.epsilon_0

# The atomic unit of mass is such that an electron has mass 1.0.
# The (confusingly named) atomic mass units are way smaller than this.
# The mass in units of electron masses is what we need to be internally consistent.
AMU2ELECTRON_MASS = 1.0 / (constants.value("atomic unit of mass") * AVOGADRO * 1000)
FS2AU = 1.0 / (constants.value("atomic unit of time") * 1e15)
KB_EhPerK = constants.value("Boltzmann constant") / constants.value("atomic unit of energy")

SYMB2Z = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88,
    "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103,
    "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112,
    "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}
