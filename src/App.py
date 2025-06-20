
from model import BeaverModel  # your adapted model
from agent import Beaver, Kit, Juvenile, Adult   # your Beaver agent class
import numpy as np
from rasterio import open as rio_open
import matplotlib.pyplot as plt

with rio_open("Users/r34093ls/Documents/test_flood/clipped_dtm.tif") as dem:  # 50m resolution
            dem = dem.read(1)

def beaver_plot (dem, agents, step=None, save_path=None):

    plt.figure(figsize = (10, 8))
    plt.imsave(dem, cmap = 'greys', origin = 'upper')

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

model = BeaverModel(initial_beavers=50, seed=42)

for i in range(120):
    model.step()

beaver_plot(model.dem, model.type[Beaver], step=12 )