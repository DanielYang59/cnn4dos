"""Plot d-band-center vs adsorption energy relationship for each adsorbate."""

import os

import yaml
from src.dBand import dBand
from src.list_dos_files import list_dos_files
from src.load_ads_energy import load_ads_energy
from src.plot_scatter import plot_scatter

if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    dos_dir = cfg["path"]["dos_dir"]
    label_dir = cfg["path"]["label_dir"]
    fermi_level_dir = cfg["path"]["fermi_level_dir"]

    adsorbates = cfg["species"]["adsorbates"]
    substrates = cfg["species"]["substrates"]
    color_dict = cfg["species"]["color_dict"]
    color_dict = dict(zip(substrates, color_dict))

    energy_range = cfg["calculation"]["energy_range"]

    for ads in adsorbates:
        # Work on all eDOS files
        d_band_centres = []
        adsorption_energies = []
        labels = []
        colors = []
        dos_files = list_dos_files(
            dos_dir,
            [
                ads,
            ],
            substrates,
        )

        for file in dos_files:
            # Calculate d-band centre
            calculator = dBand(
                dosFile=file,
                fileType="numpy",
                fermi_level_dir=fermi_level_dir,
                energy_range=energy_range,
            )

            d_band_centre = calculator.calculate_d_band_centre(
                merge_suborbitals=True, verbose=False
            )

            # Skip samples without d electron
            if d_band_centre != "NA":
                d_band_centres.append(d_band_centre)
                # Get adsorption energy
                adsorption_energies.append(
                    load_ads_energy(file, ads_energy_dir=label_dir)
                )

                labels.append(file.split(os.sep)[-4])

                # Compiles colors based on substrate
                colors.append(color_dict[file.split(os.sep)[-4]])

        # Create scatter plot
        plot_scatter(
            x=d_band_centres,
            y=adsorption_energies,
            labels=labels,
            colors=colors,
            show=False,
            savename=os.path.join("figures", f"{ads}.png"),
        )
