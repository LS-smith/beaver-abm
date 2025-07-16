
from Model import BeaverModel  # your adapted model
from Agent import Beaver, Kit, Juvenile, Adult   # your Beaver agent class
import numpy as np
from rasterio import open as rio_open
import matplotlib.pyplot as plt

import skimage.transform as skt 

def downsample (dem, max_size=1000):
     factor = max (dem.shape) / max_size
     if factor > 1:
          new_shape = (int(dem.shape[0] / factor), int(dem.shape[1] / factor))
          dem_dwn = skt.resize(dem, new_shape, anti_aliasing=True, preserve_range=True)
          return dem_dwn
     return dem

def beaver_plot (dem, agents, step=None, save_path=None):
    if not agents:
         print(f"step {step}: no agents to plot")
         return

    x_scale = dem.shape[1] / agents[0].model.dem.shape[1]
    y_scale = dem.shape[0] / agents[0].model.dem.shape[0]

    plt.figure(figsize = (10, 8))
    plt.imshow(dem, cmap = 'Grays', origin = 'upper', zorder = 1)

    count = 0
    for agent in agents:
        if getattr(agent, "remove", False):
            continue
        x, y = agent.pos
        x_down = x * x_scale
        y_down = y * y_scale
        
        if hasattr(agent,"territory"):
            territory_colour = "lightgreen"
            for tx, ty in agent.territory:
                tx_down = tx * x_scale
                ty_down = ty * y_scale
                plt.scatter(tx_down, ty_down, c=territory_colour, s=10, alpha=0.15,zorder = 2, marker = 's')

        if isinstance(agent, Kit):
            color = "green"
        elif isinstance(agent, Juvenile):
            color = "orange"
        elif isinstance(agent, Adult):
            color = "brown"
        else:
            color = "gray"
        plt.scatter(x_down, y_down, c=color, s=50,edgecolors='black', alpha=0.7,zorder = 10 ) 

        if( 0<= x_down < dem.shape[1] and
            0<= y_down < dem.shape[0] and
            dem[int (y_down), int (x_down)] != 0):
                print(f"agent at: ({y}, {x})")
                count +=1

    plt.title(f"Beaver abm{'- step' + str(step) if step is not None else ''}")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path,bbox_inches= 'tight', dpi=150)
        plt.close()
    else:
        plt.show()

    print(f"step {step}: plotted {count} agents")

with rio_open('/Users/r34093ls/Documents/GitHub/beaver-abm/data/DTM.tif') as dem:  # 5m resolution
            dem = dem.read(1)

dem_dwn = downsample(dem)

model = BeaverModel(dem=dem, initial_beavers=50, seed=42)

for i in range(120):
    model.step()
    if i % 12 == 0:
        save_path = f'/Users/r34093ls/Documents/GitHub/beaver-abm/out/gif_step_{i:03d}.png'
        beaver_plot(dem_dwn, model.type[Beaver], step=i, save_path=save_path)

