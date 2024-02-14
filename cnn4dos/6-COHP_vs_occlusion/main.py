"""Plot -COHP and occlusion experiment result (Co dxz/dxy orbitals)."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


def setup_matplotlib():
    """Sets up Matplotlib parameters."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.linewidth"] = 2


def import_data(file_path: Path):
    """Imports data from a given file path."""
    return pd.read_csv(file_path).iloc[:, [0, 1]]


def plot_data(ax, x, y, color, linewidth):
    """Plots data on a given Matplotlib axis."""
    ax.plot(x, y, color=color, linewidth=linewidth)


def main():
    setup_matplotlib()

    fermi_level = -2.06264337
    warnings.warn("Fermi level set manually.")

    # Import occlusion result
    occ_array = np.load(Path("data") / "occlusion_predictions.npy")
    occ_energy = np.linspace(-14, 6, 4000)
    occ_energy -= fermi_level
    occ_dxy = occ_array[:, 4]
    occ_dxz = occ_array[:, 7]

    # Import COHP result
    cohp_energy, cohp_dxy = import_data(Path("data") / "cohp_dxy.dat").values.T
    _, cohp_dxz = import_data(Path("data") / "cohp_dxz.dat").values.T
    cohp_dxy = -cohp_dxy
    cohp_dxz = -cohp_dxz

    # Prepare plot data
    xs = [occ_energy, occ_energy, cohp_energy, cohp_energy]
    ys = [occ_dxy, occ_dxz, cohp_dxy, cohp_dxz]
    top_color = "#F5C767"
    bottom_color = "#1984A3"

    # Create subplots
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig)
    ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i]) for i in range(4)]

    plot_data(ax1, xs[0], ys[0], top_color, 2)
    plot_data(ax2, xs[1], ys[1], top_color, 2)
    plot_data(ax3, xs[2], ys[2], bottom_color, 2)
    plot_data(ax4, xs[3], ys[3], bottom_color, 2)

    # Configure axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim([-11, 6])
        ax.tick_params(axis="both", labelsize=16, width=2.5, length=5)
        ax.yaxis.set_major_locator(MaxNLocator(5))

    # Set x/y axis labels
    fig.supxlabel("$E-E_\mathrm{f}$ (eV)", fontsize=20)
    ax1.set_ylabel("$\Delta\mathit{E}_{\mathrm{ads}}$ (eV)", fontsize=20)
    ax3.set_ylabel("-COHP", fontsize=20)

    ax1.yaxis.set_label_coords(-0.3, 0.5)
    ax3.yaxis.set_label_coords(-0.3, 0.5)

    # Adjust spacing of subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.15)

    # Save and show figure
    plt.savefig(Path("figures") / "cohp_vs_occlusion.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
