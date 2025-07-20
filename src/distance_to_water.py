import geopandas as gdp
import rasterio
from shapely.geometry import Point
import numpy as np


waterways = gdp.read_file('/Users/r34093ls/Documents/GitHub/beaver-abm/data/water_network.shp')

with rasterio.open('/Users/r34093ls/Documents/GitHub/beaver-abm/data/hsm_5m.tif') as hsm_5m:
    hsm = hsm_5m.read(1)
    transform = hsm_5m.transform

rows, cols = hsm.shape
distance_to_water = np.zeros_like(hsm, dtype=np.float32)

for row in range(rows):
    for col in range(cols):
        x,y = rasterio.transform.xy(transform, row, col)
        cell_point = Point(x,y)
        min_distance = waterways.distance(cell_point).min()
        distance_to_water[row,col] = min_distance

