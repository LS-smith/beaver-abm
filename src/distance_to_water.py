import geopandas as gdp
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, CAP_STYLE, JOIN_STYLE
import numpy as np
from scipy.ndimage import distance_transform_edt

waterways = gdp.read_file('./data/water_network.shp')

waterways = waterways[~waterways['width'].isna()].copy()

def buffer_width(row):
    return row.geometry.buffer(
        row['width'] / 2,
        cap_style = CAP_STYLE.round,
        join_style = JOIN_STYLE.round
    )
    
waterways['buffer_geom'] = waterways.apply(buffer_width, axis=1)

buff_water = waterways.set_geometry('buffer_geom')

if 'geometry' in buff_water.columns and buff_water.geometry.name != 'geometry':
    buff_water = buff_water.drop(columns = ['geometry'])

buff_water = buff_water.set_crs("epsg: 27700")
buff_water.to_file('./data/buffered_waterways.shp')

with rasterio.open('./data/hsm_5m.tif') as hsm_5m:
    hsm = hsm_5m.read(1)
    transform = hsm_5m.transform
    shape = hsm.shape
    pixel_size = hsm_5m.res[0]
    profile = hsm_5m.profile.copy()

channel_mask = rasterize(
    [(geom,1) for geom in buff_water.geometry],
    out_shape = shape,
    transform = transform,
    fill = 0,
    dtype = np.uint8,
)

distance_pixels = distance_transform_edt(channel_mask == 0)
distance_m = distance_pixels * pixel_size

profile.update(dtype=np.float32, count=1, nodata=None)
with rasterio.open ('./data/distance_to_water_5m.tif', 'w', **profile) as dtw:
                   dtw.write(distance_m.astype(np.float32), 1)

