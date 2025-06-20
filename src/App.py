from mesa.experimental.devs import ABMSimulator
from mesa.visualization import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from model import BeaverModel  # your adapted model
from agent import Beaver  # your Beaver agent class
import numpy as np
from rasterio import open as rio_open
import matplotlib.pyplot as plt

with rio_open("Users/r34093ls/Documents/test_flood/clipped_dtm.tif") as dem:  # 50m resolution
    dem = dem.read(1)

plt.imsave("dem_grid.png", dem, cmap = "greys")

def beaver_portrayal(agent):
    if not getattr(agent, "cell", None):
        return None  # skip agents with no cell

    portrayal = { "Shape": "circle",
                  "Color": "grey",
                  "Filled": "true",
                  "Layer": 2, #hmmmm
                  "r": 0.5 }
    
    if isinstance(agent, Kit):
        portrayal["color"] = "green"
        portrayal["Layer"] = 2
    elif isinstance(agent, Juvenile):
        portrayal["color"] = "orange"
        portrayal["Layer"] = 2
    elif isinstance(agent, Adult):
        portrayal["color"] = "brown"
        portrayal["Layer"] = 2
    else:
        portrayal["color"] = "gray"
        portrayal["Layer"] = 2

    return portrayal


grid = CanvasGrid(agent_portrayal,
                  dem.shape[1],
                  dem.shape[0],
                  500,
                  500,
                  background="dem_grid")



server = ModularServer(BeaverModel,
                        [grid],
                       "beaver abm",
                       {"seed": 42, "initial_beavers": 50}
                    )
server.launch()
