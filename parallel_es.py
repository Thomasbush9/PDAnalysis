import os
import sys

import numpy as np
from mpi4py import MPI

# Script to design parallel functions for Effective Strain Analysis


def parallel_setup():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size
