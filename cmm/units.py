from scipy import constants

# Using 0.529177 for BOHR2ANG and 627.51 for HARTREE2KCAL makes the comparison with Julia CMM exact.
BOHR2ANG = 0.529177 #constants.value("atomic unit of length") * 1e10
BOHR2NM = constants.value("atomic unit of length") * 1e9

ELE_CHG = constants.elementary_charge
AVOGADRO = constants.Avogadro

INV_4PI_EPS0 = 8.987551e9 * ELE_CHG * ELE_CHG * 1e7 * AVOGADRO / 4.184 # in kcal/mol * A / e^-2

HARTREE2KJ = constants.value("atomic unit of energy") * AVOGADRO / 1000
HARTREE2KCAL = 627.51 #HARTREE2KJ / 4.184

# Dipoles

# 1 Debye = 0.2081943 e*Angstrom
DEBYE2EA = 0.2081943
# 1 Debye = 0.393430 e*Bohr
DEBYE2AU = DEBYE2EA / BOHR2ANG
