"""Example: Multi-protein comparison dashboard.

Reads output CSVs from a parallel PDAnalysis run (or a combined.joblib)
and builds an interactive 2x2 HTML dashboard.

Usage:
    pip install plotly           # if not already installed
    python examples/dashboard_example.py --input_dir output/
    python examples/dashboard_example.py --joblib output/combined.joblib --metric shear
"""

import argparse
from pathlib import Path

from PDAnalysis.visualization import get_available_metrics, plot_dashboard


def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive multi-protein dashboard."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dir",
        type=str,
        help="Directory of per-protein CSV files (e.g. seq_00001.csv …)",
    )
    group.add_argument(
        "--joblib",
        type=str,
        help="Path to combined.joblib produced by the parallel pipeline.",
    )

    parser.add_argument("--metric", type=str, default="strain",
                        help="Metric column to visualise (default: strain).")
    parser.add_argument("--sort_by", type=str, default="mean",
                        choices=["mean", "max", "median", "none"],
                        help="Sort proteins by this statistic (default: mean).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: dashboard_<metric>.html).")
    parser.add_argument("--box", action="store_true",
                        help="Use box plots instead of violins.")

    args = parser.parse_args()

    source = args.input_dir or args.joblib

    # Show available metrics
    metrics = get_available_metrics(source)
    print(f"Available metrics: {', '.join(metrics)}")

    if args.metric not in metrics:
        print(f"Warning: '{args.metric}' not found in all files. "
              f"Available: {', '.join(metrics)}")
        return

    sort_by = None if args.sort_by == "none" else args.sort_by

    fig = plot_dashboard(
        source,
        metric=args.metric,
        sort_by=sort_by,
        show_violin=not args.box,
    )

    out_path = args.output or f"dashboard_{args.metric}.html"
    fig.write_html(out_path)
    print(f"Dashboard written to {out_path}")


if __name__ == "__main__":
    main()
