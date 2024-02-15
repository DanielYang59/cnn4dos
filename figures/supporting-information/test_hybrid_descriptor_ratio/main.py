"""Test and plot mixing ratio for hybrid descriptor method
for linear scaling relations.
"""


import os
import sys

import yaml

sys.path.insert(0, "../../5-volcano-plot/src/lib")

from dataLoader import dataLoader
from scalingRelation import scalingRelation
from src.plot_mixing_ratio import plot_mixing_ratio


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    adsorption_energy_path = cfg["path"]["adsorption_energy_path"]
    thermal_correction_file = cfg["path"]["thermal_correction_file"]

    substrates = cfg["species"]["substrates"]
    adsorbates = cfg["species"]["adsorbates"]

    descriptor_x = cfg["reaction"]["descriptor_x"]
    descriptor_y = cfg["reaction"]["descriptor_y"]

    # Loading adsorption energy
    loader = dataLoader()
    loader.load_adsorption_energy(adsorption_energy_path, substrates, adsorbates)

    loader.calculate_adsorption_free_energy(correction_file=thermal_correction_file)

    # Calculate adsorption energy linear scaling relations
    calculator = scalingRelation(
        adsorption_energy_dict=loader.adsorption_free_energy,
        descriptors=(descriptor_x, descriptor_y),
        mixing_ratios="AUTO",
        verbose=True,
        remove_ads_prefix=True,
    )

    # Get mixing ratio test results
    _mixing_ratio_results = calculator.mixing_ratio_results

    # Unpack mixing ratio test results to lists
    mixing_ratio_results = {ads: [] for ads in adsorbates}
    average_result = []
    for ratio in _mixing_ratio_results:
        average = []
        for ads in adsorbates:
            if ads not in {descriptor_x, descriptor_y}:
                mixing_ratio_results[ads].append(
                    _mixing_ratio_results[ratio][ads].rvalue
                )
                average.append(_mixing_ratio_results[ratio][ads].rvalue)

        average_result.append(sum(average) / len(average))

    # Plot mixing ratio test for each adsorbate
    for ads in adsorbates:
        if ads not in {descriptor_x, descriptor_y}:
            plot_mixing_ratio(
                x=range(0, 101),
                y=mixing_ratio_results[ads],
                savename=os.path.join("figures", f"mixing_{ads}.png"),
            )

    plot_mixing_ratio(
        x=range(0, 101),
        y=average_result,
        savename=os.path.join("figures", "mixing_average.png"),
    )
