from scipy import constants

# Using 0.529177 for BOHR2ANG and 627.51 for HARTREE2KCAL makes the comparison with Julia CMM exact.
BOHR2ANG = constants.value("atomic unit of length") * 1e10
BOHR2NM = constants.value("atomic unit of length") * 1e9

ELE_CHG = constants.elementary_charge
AVOGADRO = constants.Avogadro

INV_4PI_EPS0 = 8.987551e9 * ELE_CHG * ELE_CHG * 1e7 * AVOGADRO / 4.184 # in kcal/mol * A / e^-2

DEBYE2EA = 0.2081943
DEBYE2EBOHR = DEBYE2EA / BOHR2ANG

HARTREE2KJ = constants.value("atomic unit of energy") * AVOGADRO / 1000
HARTREE2KCAL = HARTREE2KJ / 4.184
HARTREE2EV = constants.value("Hartree energy in eV")

EPSILON0 = constants.epsilon_0

# The atomic unit of mass is such that an electron has mass 1.0.
# The (confusingly named) atomic mass units are way smaller than this.
# The mass in units of electron masses is what we need to be internally consistent.
AMU2ELECTRON_MASS = 1.0 / (constants.value("atomic unit of mass") * AVOGADRO * 1000)
FS2AU = 1.0 / (constants.value("atomic unit of time") * 1e15)
KB_EhPerK = constants.value("Boltzmann constant") / constants.value("atomic unit of energy")

if __name__ == '__main__':
    print(HARTREE2KCAL, HARTREE2KJ)
