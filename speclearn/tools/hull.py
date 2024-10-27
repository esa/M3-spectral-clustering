from asyncio import constants
from email.mime import base
import pysptools.spectro as spectro
import pandas as pd
from .constants import *


def display_convex_hull(spectrum, s_2=0):
    if s_2 < 0:
        wvl = list(GLOBAL_WAVELENGTH[0:s_2])
    else:
        wvl = list(GLOBAL_WAVELENGTH)

    schq = spectro.SpectrumConvexHullQuotient(spectrum, wvl)
    schq.display('display_name')


def extract_and_display_features(spectrum, baseline, display=True, s_2=0):
    if s_2 < 0:
        wvl = list(GLOBAL_WAVELENGTH[0:s_2])
    else:
        wvl = list(GLOBAL_WAVELENGTH)

    fea = spectro.FeaturesConvexHullQuotient(spectrum, wvl, baseline=baseline)
    if display:
        fea.display('display_name', feature='all')
    return fea


def extract_crs(spectrum, norm=True):
    baseline = 0
    fea = extract_and_display_features(spectrum, baseline, display=False)
    return fea.crs


def binned_to_df(binned, bins):
    data = binned
    df_2d = pd.DataFrame(
        data=data, index=bins['long_center'], columns=bins['lat_center'])
    df_2d = df_2d.sort_index(ascending=False)
    df_2d = df_2d.reindex(sorted(df_2d.columns), axis=1)
    return df_2d.T
