"""Generate a multi-protein dashboard from the Lysozyme test data.

Picks the first AlphaFold PDB as the reference structure and compares it
against all other Lysozyme PDBs, computing strain + shear for each pair.
The resulting CSVs are fed straight into the interactive dashboard.

Usage:
    python examples/dashboard_lysozyme.py
    # opens dashboard_strain.html in your browser
"""

import tempfile
from pathlib import Path

from PDAnalysis import Protein, Deformation
from PDAnalysis.visualization import get_available_metrics, plot_dashboard

DATA_DIR = Path("test_data/Lysozyme")
OUT_HTML = Path("examples/output")
OUT_HTML.mkdir(parents=True, exist_ok=True)

# Separate AlphaFold models (same length) from experimental PDBs
all_pdbs = sorted(DATA_DIR.glob("*.pdb"))
af_pdbs = [p for p in all_pdbs if p.name.startswith("AF-")]
exp_pdbs = [p for p in all_pdbs if not p.name.startswith("AF-")]

# Use AF models (all 148 residues) for the dashboard — pick first as reference
pdbs = af_pdbs if len(af_pdbs) > 1 else exp_pdbs
reference = pdbs[0]
targets = pdbs[1:]

print(f"Reference: {reference.name}")
print(f"Targets:   {[p.name for p in targets]}")

# Generate per-pair CSVs in a temp directory
csv_dir = Path(tempfile.mkdtemp(prefix="lysozyme_dashboard_"))
print(f"\nCSV output dir: {csv_dir}")

protA = Protein(str(reference), min_plddt=70)

for i, target_path in enumerate(targets, start=1):
    label = target_path.stem
    print(f"  [{i}/{len(targets)}] {reference.name} vs {label} ...", end=" ", flush=True)
    protB = Protein(str(target_path), min_plddt=70)
    deform = Deformation(protA, protB, method=["strain", "shear"])
    deform.run()
    out_csv = csv_dir / f"{label}.csv"
    deform.save_output(str(out_csv))
    print("done")

# Show available metrics
metrics = get_available_metrics(str(csv_dir))
print(f"\nAvailable metrics: {metrics}")

# Build dashboards
for metric in metrics:
    fig = plot_dashboard(str(csv_dir), metric=metric, sort_by="mean")
    html_path = OUT_HTML / f"lysozyme_dashboard_{metric}.html"
    fig.write_html(str(html_path))
    print(f"Dashboard written: {html_path}")

print("\nDone! Open the HTML files in a browser.")
