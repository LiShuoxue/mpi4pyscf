
import pytest
from pyscf import gto

def H2O_mol():
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvtz'
    mol.spin = 0
    mol.build()
    return mol


# @pytest.fixture
def H2O_trimer_mol():
    mol = gto.Mole()
    mol.atom =  f'''
    O  8.70835e-01  6.24543e+00  5.08445e+00
    H  8.51421e-01  6.33649e+00  6.05969e+00
    H  1.66206e+00  5.60635e+00  5.02479e+00
    O  2.38299e+00  8.36926e+00  4.10083e+00
    H  1.76679e+00  7.63665e+00  4.17552e+00
    H  2.40734e+00  8.80363e+00  4.99023e+00
    O  2.41917e+00  1.04168e+01  2.48601e+00
    H  2.55767e+00  9.70422e+00  3.12008e+00
    H  3.10835e+00  1.02045e+01  1.83352e+00 '''
    mol.basis = 'cc-pvtz'
    mol.build()
    # mol.nelectron = mol.nelectron + 2
    mol.build()
    return mol


# @pytest.fixture
def H2O_dimer_mol():
    mol = gto.Mole()
    mol.atom =  '''
    O  8.70835e-01  6.24543e+00  5.08445e+00
    H  8.51421e-01  6.33649e+00  6.05969e+00
    H  1.66206e+00  5.60635e+00  5.02479e+00
    O  2.38299e+00  8.36926e+00  4.10083e+00
    H  1.76679e+00  7.63665e+00  4.17552e+00
    H  2.40734e+00  8.80363e+00  4.99023e+00 '''
    mol.basis = 'cc-pvtz'
    mol.build()
    return mol
