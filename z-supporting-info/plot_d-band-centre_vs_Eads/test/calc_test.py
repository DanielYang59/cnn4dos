
import sys
sys.path.append('../src')
# mport dBand
import numpy as np
import pandas as pd


def __calculate_band_moment(single_dos_orbital, energy_array, ordinal):
    # Calculate band moment
    density = np.copy(single_dos_orbital) ** ordinal

    numerator = np.trapz(y=(density * np.copy(energy_array)),
                            dx=(energy_array[-1] - energy_array[0]) / energy_array.shape[0]
                            )

    denominator = np.trapz(y=density,
                            dx=(energy_array[-1] - energy_array[0]) / energy_array.shape[0]
                            )

    return numerator / denominator


if __name__ == "__main__":
    # Import test DataFrame
    df = pd.read_csv("d_band_test_neg1.59.csv")

    # Unpack energy and d-band data
    energy_array = np.array(df["Energy"])
    d_band_array = np.array(df["d-DOS"])


    # Calculate d-band centre
    centre = __calculate_band_moment(
        single_dos_orbital=d_band_array,
        energy_array=energy_array,
        ordinal=1,

    )

    print(centre)