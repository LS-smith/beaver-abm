

from Model import Flood_Model
from rasterio import open as rio_open

# Load DEM data
with rio_open('./data/DTM.tif') as dem_src:  # 5m resolution
    dem = dem_src.read(1)
    dem_transform = dem_src.transform

# Create and run model
model = Flood_Model(dem=dem, dem_transform=dem_transform, initial_beavers=50, seed=None)

# Run simulation for 120 time steps (10 years)
for i in range(120):
    model.step()

# Save results
model.datacollector.get_model_vars_dataframe().to_csv("./out/Beaver_data.csv")

print("Simulation complete. Data saved to beaver_data.csv :)")

