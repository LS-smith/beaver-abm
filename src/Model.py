from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.experimental.devs import ABMSimulator
import numpy as np
from rasterio import open as rio_open
import geopandas as gdp

from Agent import Beaver, Kit, Juvenile, Adult, Dam # if this is seperate files

class Flood_Model(Model):
    def __init__(self, dem, dem_transform, initial_beavers=1, seed=None, simulator=None): # initialise
        super().__init__(seed=seed)

        print("Reading HSM...")
        with rio_open('./data/hsm_5m_clip.tif') as hsm:
            self.hsm = hsm.read(1)
        print("HSM loaded bruh")

        print("Reading dtw...")
        with rio_open('./data/distance_to_water_5m_clip.tif') as dtw:
            self.distance_to_water = dtw.read(1)
        print("dtw also done...")

        print("Reading waterways...")
        self.waterways = gdp.read_file('./data/water_network_clip.shp') 
        print("waterways not the issue...")

        self.dem = dem
        self.dem_transform = dem_transform
        
        self.height, self.width = self.dem.shape

        # properly initialise the grid
        print("initialsing grid...")
        self.grid = MultiGrid(self.width, self.height, torus=True)
        print("done")
        # initialise type as a set NOT list
        self.type = {Beaver: [], Dam: []}

        print("building valid area.")
        ys, xs = np.nonzero(self.dem != 0)
        valid_area =np.column_stack((xs, ys))  
        print("DONE .")

        # create initial beavers and add them to the grid
        print("creating agents...")
        for _ in range(initial_beavers):
            x, y =self.random.choice(valid_area)
            #x = self.random.randrange(self.width)
            #y = self.random.randrange(self.height)
            beaver = Juvenile(self) # add only adult beavers (may be self.unique_id)
            self.grid.place_agent(beaver, (x,y))
            self.type[Beaver].append(beaver)
        print("agents created.")

        print("after model creation:")
        print("beavers in model.type[Beaver]:", len(self.type[Beaver]))
        #print("total number of agents in the grid:", sum(len(cell_contents) for cell_contents, pos in self.grid.coord_iter()))


        self.datacollector = DataCollector({
            "Beaver_num": lambda m: len(m.type[Beaver]),
            "Paired Beavers": lambda m: len([a for a in m.type[Beaver] if a.partner and a.unique_id < a.partner.unique_id]),
            "Males": lambda m: len([a for a in m.type[Beaver] if a.sex == "M"]),
            "Females": lambda m: len([a for a in m.type[Beaver] if a.sex == "F"]),
            "Kits": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Kit)]),
            "Juveniles": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Juvenile)]),
            "Adults": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Adult)]),
            "territory_size": lambda m: [len(b.territory) if hasattr(b, "territory") and b.territory else 0 for b in m.type[Beaver]],
            "territory_location": lambda m: [list(b.territory) if hasattr(b, "territory") and b.territory else [] for b in m.type[Beaver]],
            #"Dams": lambda m: len(m.type[Dam]),
            #"flooded_cell_count":
            #"Flooded_cell_location":
            #number of beavers in colony

        })
        self.datacollector.collect(self)

        if simulator is not None:
            self.simulator = simulator
            self.simulator.setup(self)   
        self.running = True

        self.month = 1



    def step(self):

        self.month += 1
        if self.month > 12:
            self.month = 1

        # update the agents
        for agent in list(self.type[Beaver]):
            agent.step()

        for agent in list(self.type[Beaver]):
            if getattr(agent, "remove", False):
                self.grid.remove_agent(agent)
                self.type[Beaver].remove(agent)
        
        self.datacollector.collect(self) # collect data on each step


