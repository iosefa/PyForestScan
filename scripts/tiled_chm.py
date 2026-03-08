#!/usr/bin/env python3
"""Create tiled CHM rasters from an EPT dataset using PyForestScan."""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import geopandas as gpd
import numpy as np
from numpy.lib import recfunctions as rfn

from pyforestscan.calculate import calculate_chm
from pyforestscan.handlers import create_geotiff, read_lidar
from pyforestscan.utils import get_bounds_from_ept, get_srs_from_ept

DEFAULT_EPT = "/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Lidar/ept-full/ept.json"
DEFAULT_POLYGON = "/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Puuwaawaa/Extents/Ahupuaa_Laupahoehoe/laupahoehoe.gpkg"


def ensure_height_above_ground(arr: np.ndarray) -> np.ndarray:
    """Ensure HeightAboveGround exists (required by calculate_chm)."""
    if "HeightAboveGround" in arr.dtype.names:
        return arr
    if "Z" not in arr.dtype.names:
        raise ValueError("Point cloud must contain either 'HeightAboveGround' or 'Z'.")

    # Fallback for datasets without precomputed HAG.
    hag = arr["Z"] - np.nanmin(arr["Z"])
    return rfn.append_fields(arr, "HeightAboveGround", hag, usemask=False)


def load_polygon_wkt_and_bounds(polygon_path: str) -> Tuple[str, Tuple[float, float, float, float]]:
    """Load polygon file, dissolve all features, and return WKT + bounds."""
    gdf = gpd.read_file(polygon_path)
    if gdf.empty:
        raise ValueError(f"Polygon file has no features: {polygon_path}")

    # Prefer union_all when available (GeoPandas/Shapely newer versions).
    if hasattr(gdf.geometry, "union_all"):
        geom = gdf.geometry.union_all()
    else:
        geom = gdf.geometry.unary_union

    if geom.is_empty:
        raise ValueError(f"Polygon geometry is empty after dissolve: {polygon_path}")

    min_x, min_y, max_x, max_y = geom.bounds
    return geom.wkt, (min_x, max_x, min_y, max_y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ept", default=DEFAULT_EPT, help="Path/URL to ept.json")
    parser.add_argument(
        "--polygon",
        default=DEFAULT_POLYGON,
        help="Polygon file (e.g., .gpkg) used to clip lidar reads",
    )
    parser.add_argument("--output-dir", default="chm_tiles", help="Output directory for CHM tile GeoTIFFs")
    parser.add_argument("--tile-size", type=float, default=1000.0, help="Tile size in map units (x and y)")
    parser.add_argument("--xy-res", type=float, default=0.5, help="XY voxel/grid resolution")
    parser.add_argument(
        "--interpolation",
        default="linear",
        choices=["nearest", "linear", "cubic", "none"],
        help="Interpolation method for CHM gaps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ept_min_x, ept_max_x, ept_min_y, ept_max_y, min_z, max_z = get_bounds_from_ept(args.ept)
    srs = get_srs_from_ept(args.ept)
    if not srs:
        raise ValueError("Could not determine SRS from EPT metadata.")

    polygon_wkt, (poly_min_x, poly_max_x, poly_min_y, poly_max_y) = load_polygon_wkt_and_bounds(args.polygon)
    min_x = max(ept_min_x, poly_min_x)
    max_x = min(ept_max_x, poly_max_x)
    min_y = max(ept_min_y, poly_min_y)
    max_y = min(ept_max_y, poly_max_y)

    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Polygon bounds do not overlap EPT bounds.")

    tile_w = tile_h = args.tile_size
    num_tiles_x = int(np.ceil((max_x - min_x) / tile_w))
    num_tiles_y = int(np.ceil((max_y - min_y) / tile_h))

    processed = 0
    skipped = 0
    interpolation = None if args.interpolation == "none" else args.interpolation

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            tile_min_x = min_x + i * tile_w
            tile_max_x = min(max_x, tile_min_x + tile_w)
            tile_min_y = min_y + j * tile_h
            tile_max_y = min(max_y, tile_min_y + tile_h)

            bounds = ([tile_min_x, tile_max_x], [tile_min_y, tile_max_y], [min_z, max_z])
            arrays = read_lidar(args.ept, srs=srs, bounds=bounds, crop_poly=True, poly=polygon_wkt)
            if not arrays or arrays[0].size == 0:
                skipped += 1
                continue

            points = ensure_height_above_ground(arrays[0])
            chm, extent = calculate_chm(
                points,
                voxel_resolution=(args.xy_res, args.xy_res, 1.0),
                interpolation=interpolation,
            )

            out_tif = os.path.join(args.output_dir, f"tile_{i}_{j}_chm.tif")
            create_geotiff(chm, out_tif, srs, extent)
            processed += 1

    print(f"Finished. Wrote {processed} tiles, skipped {skipped} empty tiles.")


if __name__ == "__main__":
    main()
