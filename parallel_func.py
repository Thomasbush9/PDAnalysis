import os
import sys
from typing import List
import numpy as np
from mpi4py import MPI
import pandas as pd

# Script to design parallel functions for Effective Strain Analysis


def parallel_setup():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

#TODO add the combining function 
def head_workers_queue(protA:List[str], paths:List[str], output:str, argv,  func, *args, **kwargs):
    comm, rank, size = parallel_setup()
    head = 0
    tot_n = len(paths)
    next_to_do = 0
    completed = 0
    if rank == 0:
        for dst in range(1, size):
            print(f"source sending to {dst}")
            comm.send(np.int32(next_to_do), dest=dst, tag=11)
            next_to_do += 1
            if next_to_do == tot_n:
                break

        # continue until finished to end once worker is done
        while next_to_do < tot_n:
            status = MPI.Status()
            d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            src = status.Get_source()
            tag = status.Get_tag()
            completed+=1
            comm.send(np.int32(next_to_do), dest=src, tag=11)
            next_to_do += 1
        while completed < tot_n:
            _ = comm.recv(source=MPI.ANY_SOURCE, tag=11)
            completed +=1

        for i in range(1, size):
            comm.send(np.int32(-1), dest=i, tag=11)
        # concat all the csv in one: 
        all_csv_path = [
            os.path.join(output, csv_name) for csv_name in os.listdir(output) if csv_name.endswith(".csv") and csv_name != "combined.csv"
        ]
        df_combined = pd.concat(map(pd.read_csv, all_csv_path), ignore_index=True)
        df_combined.to_csv(os.path.join(output, "combined.csv"))
    else:
        while True:
            idx = comm.recv(source=0, tag=11)
            print(f"Rank: {rank}, received {idx} from source")
            if idx == np.int32(-1):
                break
            func(protA, [paths[idx]], output, argv,  *args, **kwargs)
            comm.send(np.int32(1), dest=head, tag=11)

    MPI.Finalize()
    print("finished parallel run") 
