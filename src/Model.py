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

        print("Reading HSM...")
        with rio_open('./data/hsm_5m.tif') as hsm:
            self.hsm = hsm.read(1)
        print("HSM loaded bruh")

        print("Reading dtw...")
        with rio_open('./data/distance_to_water_5m.tif') as dtw:
            self.distance_to_water = dtw.read(1)
        print("dtw also done...")

        print("Reading waterways...")
        self.waterways = gdp.read_file('./data/Water_network.shp') 
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
        release_groups = 10
        beavers_per_group = 5

        #find all suitable cells near water
        suitable_cells = np.argwhere((self.hsm >= 2) & (self.hsm <= 4) & (self.distance_to_water < 50)   )

        for group in range(release_groups):
            center_idx = self.random.choice(range(len(suitable_cells)))
            center_y, center_x = suitable_cells[center_idx]
            for _ in range(beavers_per_group):
                angle = self.random.uniform(0, 2 * np.pi)
                radius = self.random.randint(0, 10)  # within 10 cells of center
                dx = int(radius * np.cos(angle))
                dy = int(radius * np.sin(angle))
                x = np.clip(center_x + dx, 0, self.hsm.shape[1] - 1)
                y = np.clip(center_y + dy, 0, self.hsm.shape[0] - 1)
                if self.hsm[y, x] in [2, 3, 4]:
                    beaver = Adult(self, sex=self.random.choice(["M", "F"]))
                    self.grid.place_agent(beaver, (x, y))
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
            "Territory_size": lambda m: [len(b.territory) if hasattr(b, "territory") and b.territory else 0 for b in m.type[Beaver]],
            "Territory_location": lambda m: [list(b.territory) if hasattr(b, "territory") and b.territory else [] for b in m.type[Beaver]],
            "Beavers_per_colony": lambda m: [
                {"Colony_size": sum([b.territory == t and b.territory for b in m.type[Beaver]]),
                "Territory_size": len(t),
                "Territory_location": list(t) }
                for t in set(tuple(b.territory) for b in m.type[Beaver] if b.territory)],
            "Dam_num": lambda m: len(m.type[Dam]),
            "Dam_locations": lambda m: [d.pos for d in m.type[Dam]],
            "Flooded_cells": lambda m: sum(np.sum(d.flooded_area) if hasattr(d, "flooded_area") and d.flooded_area is not None else 0
             for d in m.type[Dam]),
            "Flood_location": lambda m: [[(r, c) for r, c in zip(*np.where(d.flooded_area == 1))]
                if hasattr(d, "flooded_area") and d.flooded_area is not None else []
                for d in m.type[Dam]],

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


