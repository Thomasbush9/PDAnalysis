# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDAnalysis (Protein Deformation Analysis) is a Python library and CLI tool for calculating structural deformation between protein conformations. It accepts PDB, mmCIF, or raw coordinate files and produces per-residue deformation metrics as CSV output. Python >=3.8, pinned to 3.10 locally.

## Development Setup

```bash
# Environment uses uv (uv.lock present)
uv sync
source .venv/bin/activate

# Or install in editable mode
pip install -e .
```

## Running

```bash
# Single comparison
python main.py --protA path/A.pdb --protB path/B.pdb --min_plddt 70

# Ensemble averaging (multiple structures per protein)
python main.py --prot_listA paths_A.txt --prot_listB paths_B.txt --method strain shear

# All methods
python main.py --protA A.pdb --protB B.pdb --method all

# MPI parallel
mpiexec -n N python main.py --parallel True --protA A.pdb --path_list paths.txt --out_dir output/
```

## Testing

No test suite exists. Validate manually using `test_data/` examples:

```bash
python main.py --protA test_data/Lysozyme/AF-P61626-F1-model_v4.pdb \
               --protB test_data/Lysozyme/AF-P79180-F1-model_v4.pdb --min_plddt 70
```

## Architecture

### Data Flow

```
Input files → Protein / AverageProtein → Deformation.run() → .save_output("out.csv")
```

### Key Modules (all under `PDAnalysis/`)

- **`protein.py`** — `Protein` (single structure) and `AverageProtein` (ensemble-averaged neighborhoods). Stores alpha-carbon coords, distance matrices, and neighbor tensors. NaN-masks missing/excluded residues.
- **`deformation.py`** — `Deformation` class. Takes two protein objects, validates consistency, runs per-residue metric calculations. Methods: `strain`, `shear`, `non_affine`, `ldd`, `lddt`, `neighborhood_dist`, `rmsd`, `mut_dist`.
- **`pdb_parser.py`** — PDB/mmCIF parsing via Biopython. `load_and_fix_pdb_data()` handles SEQRES alignment for experimental structures with missing residues.
- **`utils.py`** — Kabsch rotation (SVD-based), mutation detection, shared neighbor index computation.

### Entry Points

- **`main.py`** — CLI (argparse). Dispatches to serial `main()` or parallel `main_parallel()` based on `--parallel`.
- **`parallel_func.py`** — MPI head-worker queue pattern. Rank 0 distributes work, ranks 1+ process. Outputs combined as `combined.joblib`.

### Important Design Patterns

- **Alpha-carbon only**: All calculations use CA coordinates exclusively.
- **NaN masking**: Residues with missing coords, low pLDDT, or high B-factor are NaN in `coord`. All calculations skip NaN positions.
- **Kabsch rotation**: Used for rotationally-invariant alignment in both `AverageProtein` averaging and `Deformation` metrics.
- **Shared neighbor indices**: Before computing deformation, only neighbors present in BOTH structures are used (`get_shared_indices()`).
- **kwargs pattern**: `Protein`, `AverageProtein`, and `Deformation` use `**kwargs` with `kwargs.get('key', default)` — no dataclasses.
- **pLDDT default asymmetry**: `AverageProtein` defaults `min_plddt=70.0`, while `Protein` defaults to `0.0`.

### Output Format

CSV with columns: `residue_index`, `protA_resname`, `protB_resname`, then one column per metric. RMSD produces both `rmsd_per_residue` and `rmsd_overall`.

## Git Branches

- `main` — stable release
- `upstream` — active development

## Dependencies

numpy, biopython, pandas, scipy (core); mpi4py + mpich (parallel); joblib (parallel output serialization).
