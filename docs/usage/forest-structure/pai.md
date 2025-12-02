# Plant Area Index (PAI)

## Theory

Plant Area Index (PAI) is a measure of the total plant material in a vertical column of the forest. It is calculated as the sum of the Plant Area Density (PAD) across all layers in the canopy.

$$ PAI = \sum_{i=1}^{n} PAD_{i-1,i} $$

Where:
-   \( PAI \) is the Plant Area Index.
-   \( PAD_{i-1,i} \) is the Plant Area Density between adjacent layers \( i-1 \) and \( i \).
-   \( n \) is the total number of layers in the vertical column.

PAI provides an aggregated view of plant material from the ground to the
top of the canopy by summing the PAD for each vertical layer.

## Calculating PAI

To calculate PAI, first calculate PAD, then:

```python
from pyforestscan.handlers import read_lidar
from pyforestscan.visualize import plot_metric
from pyforestscan.filters import filter_hag
from pyforestscan.calculate import assign_voxels, calculate_pad, calculate_pai

file_path = "../example_data/20191210_5QKB020880.laz"
arrays = read_lidar(file_path, "EPSG:32605", hag=True)
arrays = filter_hag(arrays)
points = arrays[0]

voxel_resolution = (5, 5, 1) 
voxels, extent = assign_voxels(points, voxel_resolution)

pad = calculate_pad(voxels, voxel_resolution[-1])
pai = calculate_pai(
    pad,
    voxel_height=voxel_resolution[-1],
    # Defaults to min_height=1.0; raise if you want to exclude very low vegetation.
    min_height=1.0,
)
plot_metric('Plant Area Index', pai, extent, metric_name='PAI', cmap='viridis', fig_size=None)
```

![pai.png](../../images/pai.png)

**Notes**
- `min_height` defaults to 1 m to mirror common PAI conventions. Increase it to ignore very near-ground returns or set lower (>=0) if you need the full profile.
- `max_height` optionally caps the integration height.
