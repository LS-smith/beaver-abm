from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.experimental.devs import ABMSimulator
import numpy as np
from rasterio import open as rio_open
import geopandas as gdp

from Agent import Beaver, Kit, Juvenile, Adult, Dam # if this is seperate files

class Flood_Model(Model):
    def __init__(self, dem, dem_transform, initial_beavers=50, seed=None, simulator=None): # initialise
        super().__init__(seed=seed)

        #print("Reading HSM...")
        with rio_open('./data/hsm_5m.tif') as hsm:
            self.hsm = hsm.read(1)
        #print("HSM loaded bruh")

        #print("Reading dtw...")
        with rio_open('./data/distance_to_water_5m.tif') as dtw:
            self.distance_to_water = dtw.read(1)
        #print("dtw also done...")

        #print("Reading waterways...")
        self.waterways = gdp.read_file('./data/Water_network.shp') 
        #print("waterways not the issue...")

        self.dem = dem
        self.dem_transform = dem_transform
        
        self.height, self.width = self.dem.shape

        # properly initialise the grid
        #print("initialsing grid...")
        self.grid = MultiGrid(self.width, self.height, torus=True)
        #print("done")
        # initialise type as a set NOT list
        self.type = {Beaver: [], Dam: []}

        #print("building valid area.")
        ys, xs = np.nonzero(self.dem != 0)
        valid_area =np.column_stack((xs, ys))  
        #print("DONE .")

        # create initial beavers and add them to the grid
        #print("creating agents...")
        num_pairs = initial_beavers // 2

        #find all suitable cells near water
        suitable_cells = np.argwhere((self.hsm >= 2) & (self.hsm <= 4) & (self.distance_to_water < 50)   )

        release_sites = self.random.sample(range(len(suitable_cells)), num_pairs)
        for idx in release_sites:
            y, x = suitable_cells[idx]
            male = Adult(self, sex="M")
            female = Adult(self, sex="F")
            male.partner = female
            female.partner = male
            self.grid.place_agent(male, (x, y))
            self.grid.place_agent(female, (x, y))
            self.type[Beaver].append(male)
            self.type[Beaver].append(female)
        #print("agents created.")

        #print("after model creation:")
        #print("beavers in model.type[Beaver]:", len(self.type[Beaver]))
        #print("total number of agents in the grid:", sum(len(cell_contents) for cell_contents, pos in self.grid.coord_iter()))


        self.datacollector = DataCollector({
            "beaver_num": lambda m: len(m.type[Beaver]),

            "paired_beavers": lambda m: len([a for a in m.type[Beaver] if a.partner and a.unique_id < a.partner.unique_id]),

            "males": lambda m: len([a for a in m.type[Beaver] if a.sex == "M"]),

            "females": lambda m: len([a for a in m.type[Beaver] if a.sex == "F"]),

            "kits": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Kit)]),

            "juveniles": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Juvenile)]),

            "adults": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Adult)]),

            "beaver_locations": lambda m: [a.pos for a in m.type[Beaver]],

            "colony_life_history": lambda m: [ {
                    "colony_size": sum(b.territory == t and b.territory for b in m.type[Beaver]),
                    "males": sum(b.territory == t and b.sex == "M" for b in m.type[Beaver]),
                    "females": sum(b.territory == t and b.sex == "F" for b in m.type[Beaver]),
                    "kits": sum(b.territory == t and isinstance(b, Kit) for b in m.type[Beaver]),
                    "juveniles": sum(b.territory == t and isinstance(b, Juvenile) for b in m.type[Beaver]),
                    "adults": sum(b.territory == t and isinstance(b, Adult) for b in m.type[Beaver]),
                }
                for t in set(tuple(b.territory) for b in m.type[Beaver] if b.territory)],

            "territory_size_cells": lambda m: [list(t) for t in set(tuple(b.territory) for b in m.type[Beaver] if b.territory)],

            "territory_size_km2": lambda m: [len(t) * ((5*5)/1e6) for t in set(tuple(b.territory) for b in m.type[Beaver] if b.territory)],

            #territory location as centroid
            "territory_location": lambda m: [(sum(xs)/len(t), sum(ys)/len(t)) if len(t) > 0 else (None, None) for t in set(tuple(b.territory) for b in m.type[Beaver] if b.territory) for xs, ys in [zip(*t)] if len(t) > 0],

            "territory_cells_location": lambda m: [list (t) for t in set(tuple(b.territory) for b in m.type[Beaver] if b.territory) ],

            "dam_num": lambda m: len(m.type[Dam]),

            "dam_locations": lambda m: [d.pos for d in m.type[Dam]],

            "flooded_cells": lambda m: sum(np.sum(d.flooded_area) if hasattr(d, "flooded_area") and d.flooded_area is not None else 0 for d in m.type[Dam]),

            "flood_cell_location": lambda m: [[(r, c) for r, c in zip(*np.where(d.flooded_area == 1))] if hasattr(d, "flooded_area") and d.flooded_area is not None else [] for d in m.type[Dam]],

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
