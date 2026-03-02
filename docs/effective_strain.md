# Effective Strain in Protein Deformation Analysis

## 1. Introduction

Effective strain quantifies local structural deformation between two protein conformations. Given two versions of the same protein — for example, a wild-type and a single-point mutant predicted by AlphaFold2 — effective strain measures how much each residue's local neighborhood has distorted, after removing rigid-body motion.

Unlike global metrics such as RMSD, effective strain is a **per-residue, rotationally invariant** measure. It captures the relative change in distances between a residue and its spatial neighbors, making it sensitive to local rearrangements that global alignment would miss.

**Key properties:**
- Defined per residue
- Rotationally invariant (uses Kabsch alignment per neighborhood)
- Normalized by neighbor distances (relative, not absolute)
- Normalized by neighbor count (intensive, not extensive)
- Zero for rigid-body motion; nonzero only for genuine local distortion

---

## 2. Pipeline Overview

```
┌──────────────────┐
│  Stage 1         │
│  Structure        │   PDB / mmCIF / coordinates
│  Parsing          │──→ Alpha-carbon coordinates + NaN mask
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Stage 2         │
│  Neighborhood     │   Cutoff-based neighbor sets
│  Definition       │──→ Neighbor indices + neighborhood tensors
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Stage 3         │
│  Shared Neighbor  │   Intersection of neighbor index sets
│  Computation      │──→ Shared neighbor indices for both structures
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Stage 4         │
│  Kabsch           │   SVD-based optimal rotation
│  Rotation         │──→ Rotationally aligned neighborhood tensor
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Stage 5         │
│  Effective Strain │   Per-neighbor relative displacement
│  Calculation      │──→ Scalar strain value per residue
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Stage 6         │
│  Ensemble         │   (Optional) Average over multiple structures
│  Averaging        │──→ Averaged neighborhood tensors
└──────────────────┘
```

---

## 3. Stage 1: Structure Parsing

**Module:** `pdb_parser.py`
**Functions:** `parse_pdb_coordinates()`, `parse_mmcif_coordinates()`, `load_and_fix_pdb_data()`

### 3.1 Alpha-Carbon Extraction

PDAnalysis uses **alpha-carbon (CA) coordinates only**. For each residue, the parser extracts:
- CA xyz coordinates → `coord` array of shape $(N, 3)$
- Residue indices → `idx`
- Single-letter amino acid sequence → `sequence`
- B-factor / pLDDT → `bfactor` / `plddt`

For PDB files, Biopython's `PDBParser` iterates over residues, skipping HETATM entries and residues without a CA atom. For mmCIF files, `MMCIF2Dict` is used with pandas filtering on `label_atom_id == 'CA'`.

### 3.2 SEQRES Alignment (Experimental Structures)

Experimental PDB structures often have missing residues (disordered regions). When `fix_pdb=True`, the parser:
1. Loads the full sequence from the SEQRES record
2. Aligns the ATOM-derived sequence to the SEQRES sequence using Biopython's `PairwiseAligner` (global mode, no mismatches allowed)
3. Maps atomic coordinates to correct SEQRES indices
4. Fills missing positions with NaN

### 3.3 NaN Masking

After parsing, residues are masked (set to NaN) based on quality thresholds:

- **pLDDT filtering** (`min_plddt > 0`): residues with `plddt <= min_plddt` → NaN
- **B-factor filtering** (`max_bfactor > 0`): residues with z-score of B-factor exceeding `max_bfactor` → NaN
- **Missing coordinates**: already NaN from parsing

The resulting `coord` array has shape $(N, 3)$ where $N$ is the full sequence length, with NaN rows for excluded/missing residues. All downstream calculations naturally skip NaN positions.

---

## 4. Stage 2: Neighborhood Definition

**Module:** `protein.py`
**Class:** `Protein`
**Methods:** `get_local_neighborhood()`, `_calculate_neighbor_tensor()`

### 4.1 Distance Matrix

The pairwise CA distance matrix is computed via `scipy.spatial.distance.cdist`:

$$D_{ij} = \| \mathbf{r}_i - \mathbf{r}_j \|_2$$

where $\mathbf{r}_i \in \mathbb{R}^3$ is the alpha-carbon coordinate of residue $i$. The matrix $D$ has shape $(N, N)$ and contains NaN for any pair involving a masked residue.

### 4.2 Neighbor Index Sets

For each residue $i$, its neighborhood $\mathcal{N}_i$ is defined as the set of residues within a cutoff distance $r_c$ (default 13.0 Å):

$$\mathcal{N}_i = \{ j \mid 0 < D_{ij} \leq r_c \text{ and } D_{ij} \text{ is finite} \}$$

The condition $D_{ij} > 0$ excludes the residue itself. The finiteness condition excludes neighbors with NaN coordinates.

### 4.3 Neighborhood Tensor

For each residue $i$, the **neighborhood tensor** $\mathbf{U}_i$ stores displacement vectors from residue $i$ to each of its neighbors:

$$\mathbf{U}_i = \begin{pmatrix} \mathbf{r}_{j_1} - \mathbf{r}_i \\ \mathbf{r}_{j_2} - \mathbf{r}_i \\ \vdots \\ \mathbf{r}_{j_k} - \mathbf{r}_i \end{pmatrix} \in \mathbb{R}^{k \times 3}$$

where $j_1, j_2, \ldots, j_k$ are the indices in $\mathcal{N}_i$ and $k = |\mathcal{N}_i|$. Each row is a vector pointing from the central residue to one of its neighbors.

---

## 5. Stage 3: Shared Neighbor Computation

**Module:** `utils.py`
**Function:** `get_shared_indices()`

When comparing two structures A and B, a residue may have different neighbor sets $\mathcal{N}_i^A$ and $\mathcal{N}_i^B$ (e.g., because a neighbor is masked in one structure but not the other, or because conformational changes move a residue across the cutoff boundary).

To ensure a fair comparison, PDAnalysis computes the **intersection** of neighbor index sets:

$$\mathcal{N}_i^{\text{shared}} = \mathcal{N}_i^A \cap \mathcal{N}_i^B$$

The function returns index arrays `i1` and `i2` such that:
- `neigh_tensor_A[i][i1]` extracts rows of the A-tensor corresponding to shared neighbors
- `neigh_tensor_B[i][i2]` extracts the corresponding rows of the B-tensor

Both sub-tensors now have the same number of rows, and row $m$ in both refers to the same neighbor residue.

---

## 6. Stage 4: Kabsch Rotation

**Module:** `utils.py`
**Function:** `rotate_points(P, Q)`

### 6.1 Problem Statement

Given two sets of corresponding points $P, Q \in \mathbb{R}^{k \times 3}$, find the rotation matrix $R$ that minimizes:

$$\| R \cdot P^T - Q^T \|_F^2$$

This is the **orthogonal Procrustes problem**, solved by the Kabsch algorithm.

### 6.2 Algorithm

1. Compute the cross-covariance matrix:

$$H = P^T Q \in \mathbb{R}^{3 \times 3}$$

2. Compute the singular value decomposition:

$$H = U \Sigma V^T$$

3. Compute the sign correction to ensure a proper rotation (det $R = +1$, not a reflection):

$$d = \det(V U^T)$$

4. Construct the rotation matrix:

$$R = V \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & d \end{pmatrix} U^T$$

5. Apply the rotation:

$$P' = (R \cdot P^T)^T$$

### 6.3 Usage in Effective Strain

For each residue $i$, the neighborhood tensor from structure B is rotated to best match the neighborhood tensor from structure A:

$$\mathbf{U}_i^{B'} = \text{rotate\_points}(\mathbf{U}_i^B[\text{shared}],\; \mathbf{U}_i^A[\text{shared}])$$

This removes rigid-body rotational differences between the two local neighborhoods, isolating genuine deformation.

**Note:** The neighborhood tensors are already centered (they are displacement vectors from the central residue), so no translation correction is needed.

---

## 7. Stage 5: Effective Strain Calculation

**Module:** `deformation.py`
**Method:** `Deformation._calculate_strain_residue()`

### 7.1 Per-Neighbor Relative Displacement

After Kabsch rotation, the strain for residue $i$ is computed from the rotated neighborhood tensors. For each shared neighbor $m$:

$$\epsilon_{i,m} = \frac{\| \mathbf{u}_{i,m}^{B'} - \mathbf{u}_{i,m}^A \|}{\| \mathbf{u}_{i,m}^A \|}$$

where:
- $\mathbf{u}_{i,m}^A$ is the displacement vector from residue $i$ to shared neighbor $m$ in structure A
- $\mathbf{u}_{i,m}^{B'}$ is the same vector in structure B after Kabsch rotation

The numerator measures how much the neighbor has moved; the denominator normalizes by the original distance, giving a **relative** (dimensionless) strain.

### 7.2 Averaging Over Neighbors

The effective strain for residue $i$ is the mean over all shared neighbors:

$$\varepsilon_i = \frac{1}{|\mathcal{N}_i^{\text{shared}}|} \sum_{m \in \mathcal{N}_i^{\text{shared}}} \epsilon_{i,m}$$

### 7.3 Complete Formula

Combining the above:

$$\boxed{\varepsilon_i = \frac{1}{|\mathcal{N}_i^{\text{shared}}|} \sum_{m=1}^{|\mathcal{N}_i^{\text{shared}}|} \frac{\left\| \text{Kabsch}(\mathbf{U}_i^B)_m - (\mathbf{U}_i^A)_m \right\|}{\left\| (\mathbf{U}_i^A)_m \right\|}}$$

### 7.4 Normalization Flags

The default behavior (relative + normalized) can be overridden:

| Flag | Effect |
|------|--------|
| `force_absolute=True` | Skip division by $\|\mathbf{u}_{i,m}^A\|$ (absolute displacement instead of relative strain) |
| `force_nonorm=True` | Skip division by $|\mathcal{N}_i^{\text{shared}}|$ (sum instead of mean) |

### 7.5 Output

The result is a 1D array of length $N$ (sequence length), where:
- Valid residues contain a non-negative float (the effective strain)
- Excluded/missing residues contain NaN

---

## 8. Stage 6: Ensemble Averaging (Optional)

**Module:** `protein.py`
**Class:** `AverageProtein`

When multiple structures are available for one or both proteins (e.g., multiple AlphaFold2 predictions), `AverageProtein` creates a consensus neighborhood representation.

### 8.1 Consolidated Neighbor Lists

For each residue $i$, only neighbors present in **all** input structures are retained:

$$\mathcal{N}_i^{\text{avg}} = \bigcap_{s=1}^{S} \mathcal{N}_i^{(s)}$$

where $S$ is the number of structures.

### 8.2 Rotation-Averaged Tensors

Neighborhood tensors from different structures are in different reference frames. To average them:

1. Choose the first structure as the reference
2. For each subsequent structure $s$, rotate its neighborhood tensor to match the reference using Kabsch rotation:

$$\mathbf{U}_i^{(s)'} = \text{rotate\_points}\!\left(\mathbf{U}_i^{(s)}[\text{consolidated}],\; \mathbf{U}_i^{(1)}[\text{consolidated}]\right)$$

3. Average across all structures:

$$\bar{\mathbf{U}}_i = \frac{1}{S} \sum_{s=1}^{S} \mathbf{U}_i^{(s)'}$$

The averaged tensor $\bar{\mathbf{U}}_i$ is then used in the standard deformation calculation.

### 8.3 pLDDT Default Asymmetry

`AverageProtein` defaults to `min_plddt=70.0`, while `Protein` defaults to `min_plddt=0.0`. This reflects the convention that ensemble-averaged structures (typically AlphaFold2 predictions) should filter low-confidence regions by default.

---

## 9. Related Deformation Methods

PDAnalysis implements several other per-residue deformation metrics, all following the same shared-neighbor pipeline (Stages 1–3):

### 9.1 Shear Strain (`shear`)

Magnitude of off-diagonal components of the Cauchy-Green strain tensor:

$$C = \frac{1}{2}\left[(\mathbf{U}^A)^{-T} (\mathbf{U}^A)^T \left(\mathbf{U}^B (\mathbf{U}^B)^T - \mathbf{U}^A (\mathbf{U}^A)^T\right) \mathbf{U}^A (\mathbf{U}^A)^{-T}\right]$$

$$\text{shear}_i = \frac{1}{2}\sum_j \left( C_{jj}^2 - (C C)_{jj} \right)$$

### 9.2 Non-Affine Strain (`non_affine`)

Residual of the least-squares fit of the deformation gradient tensor $F$:

$$\mathbf{U}^A F \approx \mathbf{U}^B$$

$$D_{\min}^2 = \sum \text{residuals of } \text{lstsq}(\mathbf{U}^A, \mathbf{U}^B)$$

Captures the **non-linear** component of deformation that cannot be described by an affine transformation.

### 9.3 Local Distance Difference (`ldd`)

L2 norm of differences in neighbor distances:

$$\text{LDD}_i = \left\| \|\mathbf{u}^B_m\| - \|\mathbf{u}^A_m\| \right\|_2$$

Does not require Kabsch rotation (uses scalar distances, not vectors).

### 9.4 Local Distance Difference Test (`lddt`)

Fraction of neighbor distance differences within threshold cutoffs (default: 0.5, 1, 2, 4 Å):

$$\text{LDDT}_i = \frac{1}{T \cdot |\mathcal{N}|} \sum_{t} \sum_m \mathbb{1}\left[\left|\|\mathbf{u}^B_m\| - \|\mathbf{u}^A_m\|\right| \leq c_t\right]$$

### 9.5 Neighborhood Distance (`neighborhood_dist`)

L2 norm of the difference between Kabsch-aligned neighborhood tensors:

$$\text{ND}_i = \left\| \text{Kabsch}(\mathbf{U}_i^B) - \mathbf{U}_i^A \right\|_F$$

### 9.6 RMSD (`rmsd`)

Global superposition using Kabsch alignment on all valid (non-NaN) CA coordinates, then per-residue deviation:

$$\text{RMSD}_{\text{per-res},i} = \|\mathbf{r}_i^{B'} - \mathbf{r}_i^A\|$$

$$\text{RMSD}_{\text{overall}} = \sqrt{\frac{1}{N_{\text{valid}}} \sum_i \|\mathbf{r}_i^{B'} - \mathbf{r}_i^A\|^2}$$

### 9.7 Mutation Distance (`mut_dist`)

Minimum CA distance from each residue to the nearest mutated position, averaged across both structures:

$$\text{mut\_dist}_i = \min_j \frac{D_{ij}^A + D_{ij}^B}{2}, \quad j \in \text{mutated positions}$$

---

## 10. Code Reference Table

| Stage | Description | Module | Function / Method | Key Lines |
|-------|-------------|--------|-------------------|-----------|
| 1 | PDB parsing (CA extraction) | `pdb_parser.py` | `parse_pdb_coordinates()` | 22–53 |
| 1 | mmCIF parsing | `pdb_parser.py` | `parse_mmcif_coordinates()` | 62–81 |
| 1 | SEQRES alignment | `pdb_parser.py` | `load_and_fix_pdb_data()` | 133–178 |
| 1 | NaN masking | `protein.py` | `Protein._update_nan_coords()` | 195–210 |
| 2 | Distance matrix | `protein.py` | `Protein._get_dist_mat()` | 213–215 |
| 2 | Neighbor indices | `protein.py` | `Protein.get_local_neighborhood()` | 218–223 |
| 2 | Neighborhood tensor | `protein.py` | `Protein._calculate_neighbor_tensor()` | 226–230 |
| 3 | Shared neighbor indices | `utils.py` | `get_shared_indices()` | 22–26 |
| 3 | Per-residue shared computation | `deformation.py` | `Deformation._get_shared_indices()` | 384–392 |
| 4 | Kabsch rotation (SVD) | `utils.py` | `rotate_points()` | 4–13 |
| 5 | Per-residue strain | `deformation.py` | `_calculate_strain_residue()` | 509–523 |
| 5 | Strain dispatch | `deformation.py` | `calculate_strain()` | 526–528 |
| 5 | Generic deformation loop | `deformation.py` | `_calculate_deformation()` | 412–430 |
| 6 | Consolidated neighbor lists | `protein.py` | `AverageProtein._consolidate_neighbor_lists()` | 449–456 |
| 6 | Rotation-averaged tensors | `protein.py` | `AverageProtein._rotate_and_average_neighbor_tensors()` | 461–485 |

---

## 11. References

1. **McBride, J. M.**, Polev, K., Abdirasulov, A., Reinharz, V., Grzybowski, B. A., & Tlusty, T. (2023). "AlphaFold2 can predict single-mutation effects." *Physical Review Letters*, 131, 218401.

2. **Kabsch, W.** (1976). "A solution for the best rotation to relate two sets of vectors." *Acta Crystallographica Section A*, 32(5), 922–923.

3. **Falk, M. L.**, & Langer, J. S. (1998). "Dynamics of viscoplastic deformation in amorphous solids." *Physical Review E*, 57(6), 7192–7205.

4. **Eckmann, J.-P.**, Rougemont, J., & Tlusty, T. (2019). "Colloquium: Proteins: The physics of amorphous evolving matter." *Reviews of Modern Physics*, 91(3), 031001.

5. **Mariani, V.**, Biasini, M., Barbato, A., & Schwede, T. (2013). "lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests." *Bioinformatics*, 29(21), 2722–2728.
