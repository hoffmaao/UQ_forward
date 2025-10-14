#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib

def load_results(csv_file):
    """
    Expects CSV with columns: time,QoI,sigma
    (could be produced at the end of uq_forward_sensitivity.py)
    """
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    times = data[:,0]
    QoIs  = data[:,1]
    sigmas= data[:,2]
    return times, QoIs, sigmas

def plot_figure7(csv_file, outdir="outputs", tag="uq"):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    times, QoIs, sigmas = load_results(csv_file)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8), sharex=True)

    # (a) QoI with ±2σ envelope
    ax1.plot(times, QoIs, color="k", label="QoI")
    ax1.fill_between(times, QoIs-2*sigmas, QoIs+2*sigmas,
                     color="gray", alpha=0.5, label="±2σ envelope")
    ax1.set_ylabel("QoI = ∫ h² dx")
    ax1.legend()
    ax1.set_title("(a) QoI and 2σ envelope")

    # (b) 2σ time series
    ax2.plot(times, 2*sigmas, color="red")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("2σ(QoI)")
    ax2.set_title("(b) 2σ values through time")

    plt.tight_layout()
    figfile = outdir / f"{tag}_figure7.png"
    plt.savefig(figfile, dpi=150)
    print(f"Saved figure to {figfile}")
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="CSV file produced by uq_forward_sensitivity.py (time,QoI,sigma)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--tag", default="uq")
    args = ap.parse_args()
    plot_figure7(args.csv, outdir=args.outdir, tag=args.tag)

if __name__ == "__main__":
    main()