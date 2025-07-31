from rasterio import open as rio_open
from rasterio.enums import Resampling
import numpy as np


with rio_open('./data/landcover.tif') as landcover:
    profile = landcover.profile

### upsample
    new_height = landcover.height * 2
    new_width = landcover.width * 2

    landcover_5m = landcover.read(
        1,
        out_shape = (new_height, new_width),
        resampling = Resampling.nearest)
    
    new_grid = landcover.transform * landcover.transform.scale (
        (landcover.width/new_width),
        (landcover.height/new_height)
    )

    unsuitable_lc = [12,13,15,16,17,18,19,20]
    moderate_lc = [4,5,6,7,8,9,10,11,21]
    suitable_lc = [2,3]
    preferred_lc = [1]
    water_lc = [14]

    def classify(habitat):
        if habitat in water_lc:
            return 5
        if habitat in preferred_lc:
            return 4
        elif habitat in suitable_lc:
            return 3
        elif habitat in moderate_lc:                                                     
            return 2
        elif habitat in unsuitable_lc:
            return 1
        elif habitat == 0: #no data (outside study area)
            return 0
        else:
            return 1
        
    vectorised_hsm = np.vectorize(classify)
    hsm_5m = vectorised_hsm(landcover_5m)

    profile.update(
        dtype=np.uint8,
        height=new_height,
        width=new_width,
        transform=new_grid,
        count=1,
        nodata=0,
    )

    with rio_open('./data/hsm_5m.tif', 'w', **profile) as dst:
        dst.write(hsm_5m.astype(np.uint8), 1)

    print("Upsampling and habitat suitibility classification complete. output saved as 'hsm_5m'")