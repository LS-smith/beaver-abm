from rasterio import open as rio_open
from rasterio.enums import Resampling
import numpy as np


with rio_open('/Users/r34093ls/Documents/GitHub/beaver-abm/data/landcover.tif') as landcover:
    profile = landcover.profile

### upsample
    new_height = landcover.height * 2
    new_width = landcover.width * 2

    landcover_5m = landcover.read(
        1,
        5m_shape = (new_height. new_width),
        upsample = resampling.nearest
    ) 

    new_grid = landcover.transform * landcover.transform.scale ()