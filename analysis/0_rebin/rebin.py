"""This code is used to rebin M3 images found in the location specified in the speclearn.tools.constants.py file. It is recommended to rebin all files before combining. """

from speclearn.io.transform.rebin import rebin_from_aoi

aoi = {
        'name': f'loc_-180_180_-90_90',
        'lat_range': [-90, 90],
        'long_range': [-180, 180],
        'd_lat': 0.05,
        'd_long': 0.05
}

pickle = rebin_from_aoi(aoi, combine_files=False, verbose=False)
