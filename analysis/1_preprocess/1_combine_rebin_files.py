"""This file bins all M3 images, if they are not already binned. It then combines all the binned files into one file, specified by the area of interest (AOI)."""

from speclearn.io.transform.rebin import rebin_from_aoi
from speclearn.io.data.aoi import get_full_map_aoi_longitude

import random

aoi = get_full_map_aoi_longitude(step_size=20)

random.shuffle(aoi)

for _aoi in aoi:
    pickle_file = rebin_from_aoi(_aoi, combine_files=True, verbose=True, overwrite=False, remove_spectral_outliers=False, remove_zeros_from_spectra=False, suffix='')

# for _aoi in aoi:
#     pickle_file = rebin_from_aoi(_aoi, combine_files=True, verbose=True, periods=['OP1B', 'OP2A', 'OP2B'], overwrite=False)

# for _aoi in aoi:
#     pickle_file = rebin_from_aoi(_aoi, combine_files=True, verbose=True, periods=['OP2C'], overwrite=False)
