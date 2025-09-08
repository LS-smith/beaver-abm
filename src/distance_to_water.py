# Distance to Water Calculation
# Calculates distance from each pixel to the nearest water body using buffered waterways and Euclidean distance transform

import geopandas as gdp
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, CAP_STYLE, JOIN_STYLE
import numpy as np
from scipy.ndimage import distance_transform_edt

# Load waterway network and filter out features without width data
waterways = gdp.read_file('./data/water_network_clip.shp')
waterways = waterways[~waterways['width'].isna()].copy()

def buffer_width(row):
    # Apply width-based buffer to waterway geometry
    return row.geometry.buffer(
        row['width'] / 2,
        cap_style=CAP_STYLE.round,
        join_style=JOIN_STYLE.round
    )

# Apply width-based buffering to create realistic water body polygons
waterways['buffer_geom'] = waterways.apply(buffer_width, axis=1)
buff_water = waterways.set_geometry('buffer_geom')

# Clean up geometry columns and set coordinate system
if 'geometry' in buff_water.columns and buff_water.geometry.name != 'geometry':
    buff_water = buff_water.drop(columns=['geometry'])

buff_water = buff_water.set_crs("epsg:27700")  # British National Grid
buff_water.to_file('./data/buffered_waterways_clip.shp')

# Load reference raster to match grid and projection
with rasterio.open('./data/hsm_5m_clip.tif') as hsm_5m:
    hsm = hsm_5m.read(1)
    transform = hsm_5m.transform
    shape = hsm.shape
    pixel_size = hsm_5m.res[0]
    profile = hsm_5m.profile.copy()

# Rasterise buffered waterways to create water mask
channel_mask = rasterize(
    [(geom, 1) for geom in buff_water.geometry],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype=np.uint8,
)

# Calculate Euclidean distance transform from water bodies
distance_pixels = distance_transform_edt(channel_mask == 0)
distance_m = distance_pixels * pixel_size

# Save distance raster
profile.update(dtype=np.float32, count=1, nodata=None)
with rasterio.open('./data/distance_to_water_5m_clip.tif', 'w', **profile) as dtw:
    dtw.write(distance_m.astype(np.float32), 1)

