#!/usr/bin/env python3
import argparse
import os
import re
import tempfile
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.coordinates.DCD import DCDWriter
from matplotlib.ticker import MultipleLocator


def get_expt_rdf():

    exp_path = os.path.join(os.path.dirname(__file__), "Ambient_water_xray_data.txt")

    # Read and keep ONLY lines that start with a number (skip comments/labels/units)
    numline = re.compile(r'^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')  # matches a numeric first token
    with open(exp_path, "r") as f:
        numeric_lines = [ln for ln in f if numline.match(ln)]

    buf = io.StringIO("".join(numeric_lines))
    exp_df = pd.read_csv(
        buf,
        sep=r"\s+",
        header=None,
        names=["Q", "I_X(Q)", "S_OO(Q)", "r", "g_OO(r)", "error"],
        engine="python",
    )
    return exp_df[['r', 'g_OO(r)']].values


def get_expt_rdf_2():

    exp_path = os.path.join(os.path.dirname(__file__), "soper2014.csv")
    df = pd.read_csv(exp_path)
    return df.values


def write_dcd_from_ase(images, pdb_path, dcd_path):
    """Write a DCD from ASE frames using PDB topology for atom count and order."""
    u_top = mda.Universe(pdb_path)  # topology only
    n_atoms_top = u_top.atoms.n_atoms
    n_atoms_traj = len(images[0])
    if n_atoms_top != n_atoms_traj:
        raise ValueError(f"Atom count mismatch: PDB has {n_atoms_top}, traj has {n_atoms_traj}.")

    with DCDWriter(dcd_path, n_atoms=n_atoms_top) as W:
        for im in images:
            # Positions (Å)
            pos = im.get_positions()  # (n,3)
            if pos.shape != (n_atoms_top, 3):
                raise ValueError("Position array shape mismatch with topology.")
            u_top.atoms.positions = pos

            # Cell -> MDAnalysis dimensions [lx, ly, lz, alpha, beta, gamma]
            a, b, c, alpha, beta, gamma = im.cell.cellpar()  # <- fixes the deprecation warning
            u_top.dimensions = np.array([a, b, c, alpha, beta, gamma], dtype=float)

            W.write(u_top.atoms)

def safe_pair_selection(u):
    """Return robust selections for C and O."""
    # Try common possibilities; first one that returns non-empty is used.
    candidates_C = ["name C", "type C", "prop element C"]
    candidates_O = ["name O", "type O", "prop element O"]

    def pick(cands):
        for sel in cands:
            try:
                ag = u.select_atoms(sel)
                if ag.n_atoms > 0:
                    return sel
            except Exception:
                pass
        return None

    selC = pick(candidates_C)
    selO = pick(candidates_O)
    if selC is None or selO is None:
        raise ValueError(
            "Failed to find selections for C/O. "
            "Check that your PDB has element info or atom names like 'C' and 'O'."
        )
    return selC, selO

def compute_interrdf(u, sel1, sel2, dr=0.02, rmax=None):
    # Choose rmax if needed: half the shortest box length across trajectory
    if rmax is None:
        mins = []
        for ts in u.trajectory:
            lx, ly, lz, a, b, g = ts.dimensions
            mins.append(0.5 * min(lx, ly, lz))
        rmax = float(min(mins))
    nbins = max(1, int(np.floor(rmax / dr)))
    rdf = InterRDF(u.select_atoms(sel1), u.select_atoms(sel2),
                   range=(0, rmax), nbins=nbins, 
                   exclude_same='residue'
        )
    rdf.run()
    return rdf.bins, rdf.rdf

def main():
    ap = argparse.ArgumentParser(description="RDFs (C–O, C–C, O–O) using MDAnalysis with PDB topology + ASE traj -> DCD.")
    ap.add_argument("traj", help="ASE trajectory file (.traj)")
    ap.add_argument("pdb", help="Initial PDB topology file (atom order must match trajectory)")
    ap.add_argument("--dr", type=float, default=0.05, help="Bin width in Å (default: 0.05)")
    ap.add_argument("--rmax", type=str, default="auto", help='Max radius in Å (default: "auto" = half shortest box length)')
    ap.add_argument("--plot", type=str, default="rdf_mda.png", help="Output plot (png/pdf/svg)")
    ap.add_argument("--prefix", type=str, default="rdf_mda", help="Prefix for CSV outputs")
    ap.add_argument("--keep-dcd", action="store_true", help="Keep intermediate DCD file")
    ap.add_argument("--title", type=str, default='Simulation')
    args = ap.parse_args()

    if args.traj.endswith('.traj'):
        # 1) Read ASE frames
        images = read(args.traj, ":")
        if len(images) == 0:
            raise RuntimeError("No frames read. Check your .traj file.")

        # 2) Make a temp DCD (or keep if requested)
        if args.keep_dcd:
            dcd_path = os.path.splitext(os.path.basename(args.traj))[0] + ".tmp.dcd"
        else:
            tmpdir = tempfile.mkdtemp(prefix="mda_rdf_")
            dcd_path = os.path.join(tmpdir, "traj.tmp.dcd")

        write_dcd_from_ase(images, args.pdb, dcd_path)

        # 3) Load Universe with PDB topology + DCD trajectory
        u = mda.Universe(args.pdb, dcd_path)
    else:
        u = mda.Universe(args.pdb, args.traj)
        dcd_path = None

    # 4) Selections (robust across PDBs)
    selO = 'name O'
    selH = 'name H1 or name H2'

    # 5) Compute RDFs
    rmax = None if args.rmax.lower() == "auto" else float(args.rmax)
    pairs = [
        (selO, selO, "O-O"),
        (selO, selH, 'O-H'),
        (selH, selH, 'H-H')
    ]
    results = {}
    for sel1, sel2, tag in pairs:
        r, g = compute_interrdf(u, sel1, sel2, dr=args.dr, rmax=rmax)
        results[tag] = (r, g)
        np.savetxt(f"{args.prefix}_{tag}.csv",
                   np.column_stack([r, g]), delimiter=",",
                   header="r_Angstrom,g(r)", comments="")
        print(f"Saved {args.prefix}_{tag}.csv")

    # 6) Plot
    expt1 = get_expt_rdf()
    expt2 = get_expt_rdf_2()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), constrained_layout=True)

    tags = [p[-1] for p in pairs]
    for i in range(3):
        axes[i].plot(results[tags[i]][0], results[tags[i]][1], label='CMM')

    axes[0].plot(expt2[:, 0], expt2[:, 1], label='Sopper 2014', linestyle='--')
    axes[0].plot(expt1[:, 0], expt1[:, 1], label='Skinner 2013', linestyle='--')
    axes[0].set_ylim(0, 3)

    axes[1].plot(expt2[:, 0], expt2[:, 3], label='Sopper 2014', linestyle='--')
    axes[1].set_ylim(0, 2.0)
    axes[2].plot(expt2[:, 0], expt2[:, 5], label='Sopper 2014', linestyle='--')
    axes[2].set_ylim(0, 2.0)

    for i in range(3):
        axes[i].set_xlim(0, 10)
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_xlabel(f'r({tags[i]}) / Angstrom')
        axes[i].set_ylabel(f'g({tags[i]})')
        axes[i].xaxis.set_major_locator(MultipleLocator(1.0)) 
        axes[i].xaxis.set_minor_locator(MultipleLocator(0.1)) 
        axes[i].yaxis.set_minor_locator(MultipleLocator(0.25)) 
        axes[i].yaxis.set_minor_locator(MultipleLocator(0.05)) 
        axes[i].tick_params(direction='in', which='both')
        
    fig.savefig(args.plot, bbox_inches="tight")
    print(f"Saved {args.plot}")

    # 7) Cleanup
    if not args.keep_dcd and dcd_path:
        try:
            os.remove(dcd_path)
        except OSError:
            pass

if __name__ == "__main__":
    main()
