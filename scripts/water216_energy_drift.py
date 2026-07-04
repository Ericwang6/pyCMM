# -*- coding: utf-8 -*-
"""
Analyze NVE energy conservation for the 216-water run.

Reads a text log file with columns::

    step    time(fs)    temperature(K)    total_energy(eV)    density(g/cm3)   time

and produces:

1. Summary statistics for the total energy (mean, std, min, max).
2. A linear fit of total energy vs time to estimate the drift rate.
3. A plot of total energy vs time (ps) with the linear fit overlaid.
"""

import argparse
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator

# eV to kJ/mol (per mole of molecules for consistency with MD conventions)
EV_TO_KJMOL = 96.4853074992579


def load_log(path):
    """Load time (fs) and total energy (eV) from a log or Slurm output file.

    This is robust to extra non-numeric lines (e.g. ``"Optimization finished"``,
    Python warnings, or arbitrary prefixes in Slurm ``.out`` files).

    Parameters
    ----------
    path
        Path to the log file. Lines starting with ``'#'`` are ignored, and only
        lines with at least five whitespace-separated columns are considered.

    Returns
    -------
    time_fs
        1D array of time values in femtoseconds.
    total_energy
        1D array of total energy values in electronvolts (caller converts to kJ/mol if needed).
    """
    time_fs_list = []
    energy_list = []

    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            # Expect at least: step, time(fs), T(K), total_energy(eV), density, ...
            if len(parts) < 5:
                continue

            try:
                t_fs = float(parts[1])
                e_tot = float(parts[3])
            except ValueError:
                # Skip lines that don't match the expected numeric format
                continue

            time_fs_list.append(t_fs)
            energy_list.append(e_tot)

    if not time_fs_list:
        raise ValueError("No valid energy lines found in '{}'".format(path))

    time_fs = np.asarray(time_fs_list, dtype=float)
    total_energy = np.asarray(energy_list, dtype=float)
    return time_fs, total_energy


def analyze_energy(time_fs, total_energy):
    """Compute basic statistics and energy drift.

    Follows the same conventions as :mod:`workspace_amoeba.amoeba_nve`:

    - Time is converted to ps for the regression.
    - A linear fit ``E(t) = slope * t + intercept`` is performed.
    - The drift rate is reported as ``slope`` in units of kJ/mol/ps.

    Parameters
    ----------
    time_fs
        Time values in femtoseconds.
    total_energy
        Total energy values in kJ/mol.

    Returns
    -------
    summary
        Dictionary with keys ``time_ps``, ``slope``, ``intercept``,
        ``fit_line``, ``mean``, ``std``, ``emin``, ``emax``, ``drift``.
    """
    time_ps = time_fs / 1000.0

    # Only analyze data after 100 ps
    mask = time_ps >= 100.0
    time_ps_sel = time_ps[mask]
    energy_sel = total_energy[mask]

    slope, intercept = np.polyfit(time_ps_sel, energy_sel, 1)
    fit_line = slope * time_ps_sel + intercept

    emin = float(energy_sel.min())
    emax = float(energy_sel.max())

    return {
        "time_ps": time_ps_sel,
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_line": fit_line,
        "mean": float(total_energy.mean()),
        "std": float(total_energy.std()),
        "emin": emin,
        "emax": emax,
        "drift": emax - emin,
    }


def print_summary(summary):
    """Print energy conservation statistics to stdout."""
    print("\n--- Energy Conservation Summary (216-water NVE) ---")
    print("  Mean total energy:  {:.6f} kJ/mol".format(summary["mean"]))
    print("  Std  total energy:  {:.6f} kJ/mol".format(summary["std"]))
    print("  Max  total energy:  {:.6f} kJ/mol".format(summary["emax"]))
    print("  Min  total energy:  {:.6f} kJ/mol".format(summary["emin"]))
    print("  Drift (max-min):    {:.6f} kJ/mol".format(summary["drift"]))
    print("  Energy drift rate:  {:.6e} kJ/mol/ps".format(summary["slope"]))


def _drift_label_latex(slope):
    """Format drift rate as LaTeX string: x.xx \\times 10^{n} kJ/mol/ps."""
    if slope == 0:
        return r"Linear fit (drift = $0$ kJ/mol/ps)"
    exponent = int(np.floor(np.log10(abs(slope))))
    mantissa = slope / (10 ** exponent)
    return r"Linear fit (drift = ${:.2f} \times 10^{{{:d}}}$ kJ/mol/ps)".format(
        mantissa, exponent
    )


def plot_energy(
    time_ps,
    total_energy,
    fit_line,
    slope,
    out_prefix="water216_energy",
    title=None,
):
    """Plot total energy vs time with a linear drift fit.

    Parameters
    ----------
    time_ps
        Time values in picoseconds.
    total_energy
        Total energy values in kJ/mol.
    fit_line
        Linear fit evaluated at ``time_ps``.
    slope
        Slope of the linear fit (energy drift rate, kJ/mol/ps).
    out_prefix
        Prefix for the output figure filenames (PNG and PDF).
    title
        Optional figure title; if None, no title is drawn.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    color_total = "#2563EB"
    color_fit = "#F59E0B"

    ax1.plot(
        time_ps,
        total_energy,
        color=color_total,
        lw=0.6,
        alpha=0.8,
        label="Total Energy",
        rasterized=True,
    )
    ax1.plot(
        time_ps,
        fit_line,
        color=color_fit,
        lw=2,
        ls="--",
        label=_drift_label_latex(slope),
    )

    ax1.set_xlim(100, 1000)
    ax1.set_xlabel("Time (ps)", fontsize=13)
    ax1.set_ylabel("Energy (kJ/mol)", fontsize=13)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(fontsize=10, loc="best", framealpha=0.9)
    if title is not None:
        ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax1.tick_params(axis="both", which="both", direction="in", labelsize=11)
    ax1.grid(True, alpha=0.3, lw=0.5)

    png_name = "{}.png".format(out_prefix)
    pdf_name = "{}.pdf".format(out_prefix)

    fig.savefig(png_name, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(pdf_name, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    print("\nFigure saved to {} and {}".format(png_name, pdf_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze NVE energy conservation from a log or Slurm output file.",
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default="water216_1fs_0.5Ax-b.log",
        help="Path to the log or Slurm output file.",
    )
    parser.add_argument(
        "--title",
        default="NVE Energy Conservation - 216 Water Molecules",
        help="Title for the energy plot.",
    )
    parser.add_argument(
        "--out-prefix",
        default="water216_1fs_0.5Ax-b_energy",
        help="Prefix for the saved plot files (PNG/PDF).",
    )

    args = parser.parse_args()

    print("Reading log file: {}".format(args.log_path))
    time_fs_arr, total_energy_arr = load_log(args.log_path)
    total_energy_arr = total_energy_arr * EV_TO_KJMOL  # convert eV -> kJ/mol
    summary_dict = analyze_energy(time_fs_arr, total_energy_arr)
    print_summary(summary_dict)

    plot_energy(
        summary_dict["time_ps"],
        total_energy_arr[time_fs_arr / 1000.0 >= 100.0],
        summary_dict["fit_line"],
        summary_dict["slope"],
        out_prefix=args.out_prefix,
    )

    print("Done.")

