"""Visualization module for PDAnalysis.

Provides interactive 3D protein structure plots and per-residue
deformation metric profiles using plotly.

Install the optional dependency with:
    pip install "PDAnalysis[viz]"
    # or simply: pip install plotly
"""

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "plotly is required for visualization. "
        "Install it with: pip install 'PDAnalysis[viz]' or pip install plotly"
    )

import numpy as np

from .protein import Protein, AverageProtein
from .utils import rotate_points


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_valid_coords(protein):
    """Extract CA coordinates and a boolean mask of valid (non-NaN) positions.

    For ``AverageProtein`` objects the coordinates of the first embedded
    ``Protein`` are used, since the averaged representation stores only
    neighborhood tensors (not full coordinate arrays).

    Returns
    -------
    coords : np.ndarray, shape (N, 3)
    valid : np.ndarray, shape (N,), dtype bool
    """
    if isinstance(protein, AverageProtein):
        coords = protein.proteins[0].coord
    else:
        coords = protein.coord
    valid = ~np.any(np.isnan(coords), axis=1)
    return coords, valid


def _get_contiguous_segments(valid_mask):
    """Find contiguous runs of ``True`` in *valid_mask*.

    Returns a list of (start, end) index pairs (inclusive start, exclusive end)
    suitable for slicing.  Each segment represents a stretch of consecutive
    valid residues; gaps (NaN positions) break segments so that backbone lines
    are not drawn across missing regions.
    """
    segments = []
    in_segment = False
    start = 0
    for i, v in enumerate(valid_mask):
        if v and not in_segment:
            start = i
            in_segment = True
        elif not v and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, len(valid_mask)))
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_backbone(protein, **kwargs):
    """3D interactive backbone trace (scatter + lines).

    Parameters
    ----------
    protein : Protein or AverageProtein
    color : str, optional
        Marker / line colour (default ``"royalblue"``).
    title : str, optional
    point_size : float, optional
        Marker size (default 3).
    line_width : float, optional
        Line width (default 2).
    label : str, optional
        Legend label for the trace.
    opacity : float, optional
        Trace opacity (default 1.0).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    color = kwargs.get("color", "royalblue")
    title = kwargs.get("title", "Protein Backbone")
    point_size = kwargs.get("point_size", 3)
    line_width = kwargs.get("line_width", 2)
    label = kwargs.get("label", "backbone")
    opacity = kwargs.get("opacity", 1.0)

    coords, valid = _get_valid_coords(protein)
    segments = _get_contiguous_segments(valid)

    fig = go.Figure()

    for idx, (s, e) in enumerate(segments):
        seg = coords[s:e]
        show_legend = idx == 0
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode="lines+markers",
            marker=dict(size=point_size, color=color, opacity=opacity),
            line=dict(width=line_width, color=color),
            name=label,
            legendgroup=label,
            showlegend=show_legend,
        ))

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        template="plotly_white",
    )
    return fig


def plot_strain(protein, strain_values, **kwargs):
    """3D backbone coloured by per-residue strain.

    Parameters
    ----------
    protein : Protein or AverageProtein
    strain_values : array-like, shape (N,)
        Per-residue metric values (e.g. ``deformation.strain``).
    cmap : str, optional
        Plotly colour scale name (default ``"Viridis"``).
    vmin, vmax : float, optional
        Colour scale bounds.  Defaults to data min/max.
    colorbar_label : str, optional
        Label for the colour bar (default ``"Effective Strain"``).
    nan_color : str, optional
        Colour for NaN / masked residues (default ``"lightgray"``).
    title : str, optional
    point_size : float, optional
        Marker size (default 4).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cmap = kwargs.get("cmap", "Viridis")
    colorbar_label = kwargs.get("colorbar_label", "Effective Strain")
    nan_color = kwargs.get("nan_color", "lightgray")
    title = kwargs.get("title", "Strain Map")
    point_size = kwargs.get("point_size", 4)

    coords, valid = _get_valid_coords(protein)
    strain = np.asarray(strain_values, dtype=float)

    # Determine colour bounds from valid, finite strain values
    finite_mask = valid & np.isfinite(strain)
    vmin = kwargs.get("vmin", float(np.nanmin(strain[finite_mask])) if finite_mask.any() else 0)
    vmax = kwargs.get("vmax", float(np.nanmax(strain[finite_mask])) if finite_mask.any() else 1)

    segments = _get_contiguous_segments(valid)
    fig = go.Figure()

    # Backbone lines per segment (thin, gray)
    for s, e in segments:
        seg = coords[s:e]
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode="lines",
            line=dict(width=1, color="gray"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # NaN residues (if any valid coord but NaN strain)
    nan_strain_mask = valid & ~np.isfinite(strain)
    if nan_strain_mask.any():
        c = coords[nan_strain_mask]
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers",
            marker=dict(size=point_size, color=nan_color, opacity=0.5),
            name="NaN / excluded",
            showlegend=True,
        ))

    # Coloured markers for valid strain
    if finite_mask.any():
        c = coords[finite_mask]
        sv = strain[finite_mask]
        residue_idx = np.where(finite_mask)[0]
        hover = [f"Residue {i}<br>Strain: {v:.4f}" for i, v in zip(residue_idx, sv)]
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=sv,
                colorscale=cmap,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title=colorbar_label),
                opacity=1.0,
            ),
            text=hover,
            hoverinfo="text",
            name="strain",
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        template="plotly_white",
    )
    return fig


def plot_comparison(proteinA, proteinB, **kwargs):
    """Overlay two backbones, optionally Kabsch-aligned.

    When *strain_values* is provided the markers on **both** backbones are
    coloured by the per-residue strain (shared colour scale + colorbar),
    while the backbone lines keep the A/B colours so you can still
    distinguish the two structures.

    Parameters
    ----------
    proteinA, proteinB : Protein or AverageProtein
    strain_values : array-like or None, optional
        Per-residue metric values (e.g. ``deformation.strain``).
    color_A, color_B : str, optional
        Colours for the two structures (default ``"royalblue"`` / ``"crimson"``).
        Used for lines; ignored for markers when *strain_values* is given.
    label_A, label_B : str, optional
        Legend labels (default ``"Protein A"`` / ``"Protein B"``).
    align : bool, optional
        If True (default), Kabsch-align B onto A using shared valid positions.
    cmap : str, optional
        Plotly colour scale for strain colouring (default ``"Viridis"``).
    colorbar_label : str, optional
        Label for the colour bar (default ``"Effective Strain"``).
    title : str, optional
    point_size : float, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    color_A = kwargs.get("color_A", "royalblue")
    color_B = kwargs.get("color_B", "crimson")
    label_A = kwargs.get("label_A", "Protein A")
    label_B = kwargs.get("label_B", "Protein B")
    align = kwargs.get("align", True)
    title = kwargs.get("title", "Structure Comparison")
    point_size = kwargs.get("point_size", 3)
    cmap = kwargs.get("cmap", "Viridis")
    colorbar_label = kwargs.get("colorbar_label", "Effective Strain")

    strain = None
    strain_values = kwargs.get("strain_values", None)
    if strain_values is not None:
        strain = np.asarray(strain_values, dtype=float)

    coordsA, validA = _get_valid_coords(proteinA)
    coordsB, validB = _get_valid_coords(proteinB)

    if align:
        shared_valid = validA & validB
        if shared_valid.any():
            cA = coordsA[shared_valid]
            cB = coordsB[shared_valid]
            # Centre both on shared valid positions
            cenA = cA.mean(axis=0)
            cenB = cB.mean(axis=0)
            cA_c = cA - cenA
            cB_c = cB - cenB
            # Compute rotation matrix from shared subset (Kabsch)
            H = cB_c.T @ cA_c
            U, S, Vt = np.linalg.svd(H)
            V = Vt.T
            D = np.linalg.det(V @ U.T)
            E = np.diag([1, 1, D])
            R = V @ E @ U.T
            # Apply rotation to all of B
            coordsB = ((R @ (coordsB - cenB).T).T) + cenA

    # Pre-compute shared colour bounds when strain is provided
    if strain is not None:
        finite_mask = (validA | validB) & np.isfinite(strain)
        vmin = float(np.nanmin(strain[finite_mask])) if finite_mask.any() else 0
        vmax = float(np.nanmax(strain[finite_mask])) if finite_mask.any() else 1

    fig = go.Figure()

    shown_colorbar = False
    for prot_coords, valid, line_color, label in [
        (coordsA, validA, color_A, label_A),
        (coordsB, validB, color_B, label_B),
    ]:
        segments = _get_contiguous_segments(valid)

        # Backbone lines (always use A/B colour)
        for idx, (s, e) in enumerate(segments):
            seg = prot_coords[s:e]
            fig.add_trace(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line=dict(width=2, color=line_color),
                name=label,
                legendgroup=label,
                showlegend=(idx == 0),
            ))

        # Markers — coloured by strain when available, else by A/B colour
        if strain is not None:
            finite = valid & np.isfinite(strain)
            if finite.any():
                c = prot_coords[finite]
                sv = strain[finite]
                residue_idx = np.where(finite)[0]
                hover = [f"Residue {i+1}<br>Strain: {v:.4f}" for i, v in zip(residue_idx, sv)]
                marker_dict = dict(
                    size=point_size,
                    color=sv,
                    colorscale=cmap,
                    cmin=vmin,
                    cmax=vmax,
                    opacity=1.0,
                )
                if not shown_colorbar:
                    marker_dict["colorbar"] = dict(title=colorbar_label)
                    shown_colorbar = True
                fig.add_trace(go.Scatter3d(
                    x=c[:, 0], y=c[:, 1], z=c[:, 2],
                    mode="markers",
                    marker=marker_dict,
                    text=hover,
                    hoverinfo="text",
                    name=f"{label} strain",
                    legendgroup=label,
                    showlegend=False,
                ))
        else:
            valid_coords = prot_coords[valid]
            residue_idx = np.where(valid)[0]
            hover = [f"Residue {i+1}" for i in residue_idx]
            fig.add_trace(go.Scatter3d(
                x=valid_coords[:, 0], y=valid_coords[:, 1], z=valid_coords[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=line_color),
                text=hover,
                hoverinfo="text",
                name=label,
                legendgroup=label,
                showlegend=False,
            ))

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        template="plotly_white",
    )
    return fig


def plot_side_by_side(proteinA, proteinB, strain_values=None, **kwargs):
    """Two 3D subplots side-by-side.

    If *strain_values* is provided, both subplots are coloured by strain;
    otherwise plain backbone traces are shown.

    Parameters
    ----------
    proteinA, proteinB : Protein or AverageProtein
    strain_values : array-like or None, optional
        Per-residue metric values.
    titles : tuple of str, optional
        Subplot titles (default ``("Protein A", "Protein B")``).
    cmap : str, optional
    point_size : float, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    titles = kwargs.get("titles", ("Protein A", "Protein B"))
    cmap = kwargs.get("cmap", "Viridis")
    point_size = kwargs.get("point_size", 4)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=titles,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    )

    for col, protein in enumerate([proteinA, proteinB], start=1):
        coords, valid = _get_valid_coords(protein)
        segments = _get_contiguous_segments(valid)
        scene = f"scene{col}" if col > 1 else "scene"

        # Backbone lines
        for s, e in segments:
            seg = coords[s:e]
            fig.add_trace(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line=dict(width=1, color="gray"),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=col)

        # Markers
        if strain_values is not None:
            strain = np.asarray(strain_values, dtype=float)
            finite_mask = valid & np.isfinite(strain)
            if finite_mask.any():
                c = coords[finite_mask]
                sv = strain[finite_mask]
                fig.add_trace(go.Scatter3d(
                    x=c[:, 0], y=c[:, 1], z=c[:, 2],
                    mode="markers",
                    marker=dict(
                        size=point_size,
                        color=sv,
                        colorscale=cmap,
                        colorbar=dict(title="Strain", x=1.0 if col == 2 else -0.05),
                    ),
                    showlegend=False,
                ), row=1, col=col)
        else:
            color = "royalblue" if col == 1 else "crimson"
            valid_coords = coords[valid]
            fig.add_trace(go.Scatter3d(
                x=valid_coords[:, 0], y=valid_coords[:, 1], z=valid_coords[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=color),
                showlegend=False,
            ), row=1, col=col)

        fig.layout[scene].update(aspectmode="data")

    fig.update_layout(template="plotly_white")
    return fig


def plot_strain_profile(deformation, metric="strain", **kwargs):
    """2D line chart of per-residue deformation values.

    Parameters
    ----------
    deformation : Deformation
        A ``Deformation`` object that has already been ``.run()``.
    metric : str, optional
        Name of the metric attribute on *deformation* (default ``"strain"``).
    highlight_mutations : bool, optional
        If True (default), mark mutation positions with vertical lines.
    mutation_color : str, optional
        Colour for mutation markers (default ``"red"``).
    title : str, optional
    ylabel : str, optional
        Y-axis label (default is the metric name).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    highlight_mutations = kwargs.get("highlight_mutations", True)
    mutation_color = kwargs.get("mutation_color", "red")
    title = kwargs.get("title", f"Per-Residue {metric.replace('_', ' ').title()}")
    ylabel = kwargs.get("ylabel", metric)

    values = getattr(deformation, metric)
    residue_indices = np.arange(len(values)) + 1  # 1-indexed

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=residue_indices,
        y=values,
        mode="lines",
        line=dict(color="royalblue", width=1.5),
        name=metric,
    ))

    # Highlight mutation positions
    if highlight_mutations and deformation.sub_pos is not None and len(deformation.sub_pos):
        for pos in deformation.sub_pos:
            fig.add_vline(
                x=pos + 1,  # 1-indexed
                line_dash="dash",
                line_color=mutation_color,
                opacity=0.6,
            )
        # Add invisible scatter for legend entry
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=mutation_color, dash="dash"),
            name="mutation site",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Residue Index",
        yaxis_title=ylabel,
        template="plotly_white",
    )
    return fig
