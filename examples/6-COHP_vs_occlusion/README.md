# Plot -COHP and Occlusion Experiment Results

This Python script is designed to plot -COHP (Crystal Orbital Hamilton Population) and occlusion experiment results. The script focuses on Co dxz/dxy orbitals.

## Prerequisites

- Python 3.x
- Matplotlib
- NumPy
- pandas

## File Structure

Your working directory should have the following structure:

\```
├── data
│   ├── occlusion.npy
│   ├── cohp_dxy.dat
│   └── cohp_dxz.dat
├── figures  (output directory)
└── script_name.py  (the main Python script)
\```

## Usage

1. Make sure you are in this directory.
2. Run the script:

\```bash
python3 main.py
\```

This will generate a plot in the `figures` directory named `cohp_vs_occlusion.png`.

## Description of Functions

- `setup_matplotlib()`: Sets up Matplotlib parameters such as fonts and linewidth.
- `import_data(file_path: Path)`: Reads data from a given `.dat` file and returns it as a pandas DataFrame.
- `plot_data(ax, x, y, color, linewidth)`: Plots data on a given Matplotlib axis.

## Configuration

- The Fermi level (`fermi_level`) is set manually within the script. Make sure to check this value before running the script.
