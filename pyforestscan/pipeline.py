

def _crop_polygon(polygon_wkt):
    return {
        "type": "filters.crop",
        "polygon": polygon_wkt
    }


def _hag_delaunay():
    return {
        "type": "filters.hag_delaunay"
    }


def _hag_raster(raster):
    return {
        "type": "filters.hag_dem",
        "raster": raster
    }


def _filter_hag(lower_limit=0, upper_limit=None):
    if upper_limit is None:
        limits = f"HeightAboveGround[{lower_limit}:]"
    else:
        limits = f"HeightAboveGround[{lower_limit}:{upper_limit}]"

    return {
        "type": "filters.range",
        "limits": limits
    }


def _filter_radius(radius):
    return {
        "type": "filters.sample",
        "radius": radius
    }
