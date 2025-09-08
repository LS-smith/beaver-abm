# Habitat Suitability Model Preprocessing
# Upsamples landcover data from 10m to 5m resolution and classifies into beaver habitat suitability categories (1-5 scale)

from rasterio import open as rio_open
from rasterio.enums import Resampling
import numpy as np

# Load landcover data
with rio_open('./data/landcover_clip.tif') as landcover:
    profile = landcover.profile

    # Resample from 10m to 5m resolution using nearest neighbour
    new_height = landcover.height * 2
    new_width = landcover.width * 2

    landcover_5m = landcover.read(
        1,
        out_shape=(new_height, new_width),
        resampling=Resampling.nearest
    )
    
    new_grid = landcover.transform * landcover.transform.scale(
        (landcover.width / new_width),
        (landcover.height / new_height)
    )

    # Habitat suitability classifications based on landcover types
    unsuitable_lc = [12, 13, 15, 16, 17, 18, 19, 20]  
    moderate_lc = [4, 5, 6, 7, 8, 9, 10, 11, 21]      
    suitable_lc = [2, 3]                              
    preferred_lc = [1]                               
    water_lc = [14]                                   

    def classify(habitat):
        # Classify landcover codes into beaver habitat suitability index
        if habitat in water_lc:
            return 5      # Water - highest suitability
        if habitat in preferred_lc:
            return 4      # Preferred habitat
        elif habitat in suitable_lc:
            return 3      # Suitable habitat
        elif habitat in moderate_lc:                                                     
            return 2      # Moderate suitability
        elif habitat in unsuitable_lc:
            return 1      # Unsuitable habitat
        elif habitat == 0:  # No data (outside study area)
            return 0
        else:
            return 1      # Default to unsuitable
        
    # Apply classification to entire raster
    vectorised_hsm = np.vectorize(classify)
    hsm_5m = vectorised_hsm(landcover_5m)

    # Update profile for output raster
    profile.update(
        dtype=np.uint8,
        height=new_height,
        width=new_width,
        transform=new_grid,
        count=1,
        nodata=0,
    )

    # Save habitat suitability model
    with rio_open('./data/hsm_5m_clip.tif', 'w', **profile) as dst:
        dst.write(hsm_5m.astype(np.uint8), 1)

    print("Resampling and habitat suitability classification complete. Output saved as 'hsm_5m'")