from __future__ import annotations
import os
import sys
from typing import List
import numpy as np
from mpi4py import MPI
import pandas as pd
import joblib
from pathlib import Path

# Script to design parallel functions for Effective Strain Analysis
def save_df_dict_joblib(df_dict: dict[str, pd.DataFrame], path: str | os.PathLike) -> None:
    joblib.dump(df_dict, path, compress=3)  # compress=0..9

def load_df_dict_joblib(path: str | os.PathLike) -> dict[str, pd.DataFrame]:
    return joblib.load(path)

def load_csv_dict(output_dir: str | os.PathLike, *, glob_pattern: str = "*.csv") -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    csv_dict: dict[str, pd.DataFrame] = {}

    for p in sorted(output_dir.glob(glob_pattern)):
        if p.name == "combined.csv":
            continue
        key = p.stem  # "seq_01812"
        csv_dict[key] = pd.read_csv(p)

    return csv_dict

def parallel_setup():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

#TODO add the combining function 
def head_workers_queue(protA:List[str], paths:List[str], output:Path, argv,  func, *args, **kwargs):
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
        #combine csv and save them 
        csv_dict = load_csv_dict(output)
        save_df_dict_joblib(csv_dict, os.path.join(output,"combined.joblib"))
    

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
