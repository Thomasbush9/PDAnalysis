"""Example: Visualize GFP deformation with PDAnalysis.

Compares a wild-type and mutant GFP structure, computes effective strain,
and generates five interactive HTML plots.

Usage:
    pip install plotly          # if not already installed
    python examples/visualize_gfp.py
"""

from pathlib import Path

from PDAnalysis import Protein, Deformation
from PDAnalysis.visualization import (
    plot_backbone,
    plot_strain,
    plot_comparison,
    plot_side_by_side,
    plot_strain_profile,
)

# Paths (relative to repo root)
DATA_DIR = Path("test_data/GFP")
OUT_DIR = Path("examples/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load structures ---
print("Loading structures...")
wt = Protein(str(DATA_DIR / "GFP_WT.pdb"), min_plddt=70)
mut = Protein(str(DATA_DIR / "GFP_mutant.pdb"), min_plddt=70)

# --- Compute effective strain ---
print("Computing effective strain...")
deform = Deformation(wt, mut, method="strain")
deform.run()

# --- Generate plots ---
print("Generating plots...")

fig1 = plot_backbone(wt, title="GFP Wild-Type Backbone")
fig1.write_html(str(OUT_DIR / "gfp_wt_backbone.html"))
print(f"  -> {OUT_DIR / 'gfp_wt_backbone.html'}")

fig2 = plot_strain(wt, deform.strain, title="GFP Effective Strain")
fig2.write_html(str(OUT_DIR / "gfp_strain.html"))
print(f"  -> {OUT_DIR / 'gfp_strain.html'}")

fig3 = plot_comparison(wt, mut, align=True, strain_values=deform.strain,
                       title="GFP WT vs Mutant Overlay")
fig3.write_html(str(OUT_DIR / "gfp_overlay.html"))
print(f"  -> {OUT_DIR / 'gfp_overlay.html'}")

fig4 = plot_side_by_side(wt, mut, strain_values=deform.strain,
                         titles=("Wild-Type", "Mutant"))
fig4.write_html(str(OUT_DIR / "gfp_side_by_side.html"))
print(f"  -> {OUT_DIR / 'gfp_side_by_side.html'}")

fig5 = plot_strain_profile(deform, metric="strain",
                           title="GFP Per-Residue Effective Strain")
fig5.write_html(str(OUT_DIR / "gfp_profile.html"))
print(f"  -> {OUT_DIR / 'gfp_profile.html'}")

print("\nDone! Open the HTML files in a browser to view interactive plots.")
