
from Model import BeaverModel  # your adapted model
from Agent import Beaver, Kit, Juvenile, Adult   # your Beaver agent class
import numpy as np
from rasterio import open as rio_open
import matplotlib.pyplot as plt

import skimage.transform as skt #pip install scikit-image

def downsample (dem, max_size=500):
     factor = max (dem.shape) / max_size
     if factor > 1:
          new_shape = (int(dem.shape[0] / factor), int(dem.shape[1] / factor))
          dem_dwn = skt.resize(dem, new_shape, anti_aliasing=True, preserve_range=True)
          return dem_dwn
     return dem
     

def beaver_plot (dem, agents, step=None, save_path=None):

    plt.figure(figsize = (10, 8))
    plt.imshow(dem, cmap = 'greys', origin = 'upper')

    for agent in agents:
        y, x = agent.cell.location
        if isinstance(agent, Kit):
            color = "green"
        elif isinstance(agent, Juvenile):
            color = "orange"
        elif isinstance(agent, Adult):
            color = "brown"
        else:
            color = "gray"
        plt.scatter(x, y, c= color,s=20,edgecolors='black', alpha=0.7 ) 

    plt.title(f"Beaver abm{'- step' + str(step) if step is not None else ''}")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path,bbox_inches= 'tight', dpi=150)
        plt.close()
    else:
        plt.show()

with rio_open('/Users/r34093ls/Documents/GitHub/beaver-abm/data/Clipped_dtm.tif') as dem:  # 50m resolution
            dem = dem.read(1)

dem_dwn = downsample(dem)

model = BeaverModel(initial_beavers=50, seed=42)

for i in range(120):
    model.step()
    if i % 12 == 0:
        save_path = f'/Users/r34093ls/Documents/GitHub/beaver-abm/out/gif_step_{i:03d}.png'
        beaver_plot(dem_dwn, model.type[Beaver], step=i, save_path=save_path)