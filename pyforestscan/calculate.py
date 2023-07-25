import pandas as pd
import numpy as np


def assign_voxels(arr, voxel_resolution, z_resolution):
    laz_df = pd.DataFrame(arr)

    x_bin = np.arange(laz_df['X'].min(), laz_df['X'].max() + voxel_resolution, voxel_resolution)
    y_bin = np.arange(laz_df['Y'].min(), laz_df['Y'].max() + voxel_resolution, voxel_resolution)
    z_bin = np.arange(laz_df['HeightAboveGround'].min(), laz_df['HeightAboveGround'].max() + z_resolution, z_resolution)

    histogram, edges = np.histogramdd(np.vstack((arr['X'], arr['Y'], arr['HeightAboveGround'])).transpose(),
                                      bins=(x_bin, y_bin, z_bin))

    spatial_extent = {'xmin': arr['X'].min(), 'ymin': arr['Y'].min(),
                      'xmax': arr['X'].max(), 'ymax': arr['Y'].max()}

    return histogram, spatial_extent


def calc_lad(voxel_returns, voxel_height, beer_lambert_constant=None):
    return_accum = np.cumsum(voxel_returns[::-1], axis=2)[::-1]
    shots_in = return_accum
    shots_through = return_accum - voxel_returns

    division_result = np.divide(shots_in, shots_through, out=np.full(shots_in.shape, np.nan),
                                where=~np.isnan(shots_through))

    k = beer_lambert_constant if beer_lambert_constant else 1
    dz = voxel_height

    lad = np.log(division_result) * (1 / (k * dz))

    lad = np.where(np.isinf(lad) | np.isnan(lad), np.nan, lad)

    return lad


def calc_lai(lad):
    return np.nansum(lad, axis=2)
