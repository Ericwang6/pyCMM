from scipy import constants


BOHR2ANG = constants.value("atomic unit of length") * 1e10
BOHR2NM = constants.value("atomic unit of length") * 1e9

ELE_CHG = constants.elementary_charge
AVOGADRO = constants.Avogadro

INV_4PI_EPS0 = 8.987551e9 * ELE_CHG * ELE_CHG * 1e7 * AVOGADRO / 4.184 # in kcal/mol * A / e^-2

DEBYE2EA = 0.2081943

HARTREE2KJ = constants.value("atomic unit of energy") * AVOGADRO / 1000
HARTREE2KCAL = HARTREE2KJ / 4.184


if __name__ == '__main__':
    print(HARTREE2KCAL, HARTREE2KJ)