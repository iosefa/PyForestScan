from pyforestscan.handlers import _build_pdal_pipeline
from pyforestscan.pipeline import _filter_hag


def filter_hag(arrays, lower_limit=0, upper_limit=None):
    pipeline = _build_pdal_pipeline(arrays, [_filter_hag(lower_limit, upper_limit)])

    return pipeline.arrays
