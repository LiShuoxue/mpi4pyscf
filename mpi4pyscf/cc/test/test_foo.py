"""
from mpi4pyscf.tools import mpi

rank = mpi.rank

@mpi.parallel_call
def test_foo(dev):
    print("In the test function: running rank {0}".format(rank))

if __name__ == '__main__':
    test_foo(None)
"""

from mpi4pyscf.tools import mpi



@mpi.parallel_call
def test_foo(dev):
    pass

if __name__ == '__main__':
    test_foo(None)
