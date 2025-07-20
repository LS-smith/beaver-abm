import geopandas as gdp
import rasterio
from shapely.geometry import Point
import numpy as np

waterways = gdp.read_file('/Users/r34093ls/Documents/GitHub/beaver-abm/data/water_network.shp')

default_width = 1

def buffer_width(row):
    if 'width' in row and not np.isnan(row['width']):
        return row['width'] / 2
    else:
        return default_width / 2
    
waterways['buffer_geom'] = waterways.apply(lambda row: row.geometry.buffer(buffer_width(row)), axis=1)
buff_water = waterways.set_geometry('buffer_geom')
buff_water = buff_water.drop(columns=['geometry'])

buff_water.to_file('/Users/r34093ls/Documents/GitHub/beaver-abm/data/buff_water.shp')

with rasterio.open('/Users/r34093ls/Documents/GitHub/beaver-abm/data/hsm_5m.tif') as hsm_5m:
    hsm = hsm_5m.read(1)
    transform = hsm_5m.transform

rows, cols = hsm.shape
distance_to_water = np.zeros_like(hsm, dtype=np.float32)

for row in range(rows):
    for col in range(cols):
        x,y = rasterio.transform.xy(transform, row, col)
        cell_point = Point(x,y)
        min_distance = buff_water.distance(cell_point).min()
        distance_to_water[row,col] = min_distance

(abm) (base) r34093ls@E-OSXV42GV beaver-abm % /opt/miniconda3/envs/abm/bin/python /Users/r34093ls/Documents/GitHub/beaver-abm/src/distance_to_water.py
/opt/miniconda3/envs/abm/lib/python3.13/site-packages/pyogrio/raw.py:198: UserWarning: Measured (M) geometry types are not supported. Original type 'Measured 3D LineString' is converted to 'LineString Z'
  return ogr_read(
/opt/miniconda3/envs/abm/lib/python3.13/site-packages/shapely/constructive.py:180: RuntimeWarning: invalid value encountered in buffer
  return lib.buffer(
/Users/r34093ls/Documents/GitHub/beaver-abm/src/distance_to_water.py:20: UserWarning: Column names longer than 10 characters will be truncated when saved to ESRI Shapefile.
  buff_water.to_file('/Users/r34093ls/Documents/GitHub/beaver-abm/data/buff_water.shp')
/opt/miniconda3/envs/abm/lib/python3.13/site-packages/pyogrio/geopandas.py:662: UserWarning: 'crs' was not provided.  The output dataset will not have projection information defined and may not be usable in other systems.
  write(
