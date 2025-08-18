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
        #ys, xs = np.nonzero(self.dem != 0)
        # valid_area =np.column_stack((xs, ys))  
        #print("DONE .")

        # create initial beavers and add them to the grid
        #print("creating agents...")
        num_pairs = initial_beavers // 2

        #find all suitable cells near water
        suitable_cells = np.argwhere((self.hsm >= 2) & (self.hsm <= 4) & (self.distance_to_water < 50) & (self.dem != 0)   )

        used_indices = set()
        pairs_placed = 0

        while pairs_placed < num_pairs:
            idx1 = self.random.choice(range(len(suitable_cells)))
            if idx1 in used_indices:
                continue
            y1, x1 = suitable_cells[idx1]
            # find an adjacent suitable cell not already used
            neighbors = set(
                (y1 + dy, x1 + dx)
                for dy in [-1, 0, 1]
                for dx in [-1, 0, 1]
                if not (dy == 0 and dx == 0)
            )
            neighbor_indices = [
                i for i, (y, x) in enumerate(suitable_cells)
                if (y, x) in neighbors and i not in used_indices
            ]
            if neighbor_indices:
                idx2 = self.random.choice(neighbor_indices)
                y2, x2 = suitable_cells[idx2]
                male = Adult(self, sex="M")
                female = Adult(self, sex="F")
                male.partner = female
                female.partner = male
                female.breeding_month = self.random.choice([4, 5, 6])
                female.kits_this_year = False
                self.grid.place_agent(male, (x1, y1))
                self.grid.place_agent(female, (x2, y2))
                self.type[Beaver].append(male)
                self.type[Beaver].append(female)
                used_indices.update([idx1, idx2])
                pairs_placed += 1
            else:
                used_indices.add(idx1)

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

            "colony_life_history": lambda m: [
                {"colony_size": sum(sorted(b.territory) == list(t) for b in m.type[Beaver] if b.territory),
                 "males": sum(sorted(b.territory) == list(t) and b.sex == "M" for b in m.type[Beaver] if b.territory),
                 "females": sum(sorted(b.territory) == list(t) and b.sex == "F" for b in m.type[Beaver] if b.territory),
                 "kits": sum(sorted(b.territory) == list(t) and isinstance(b, Kit) for b in m.type[Beaver] if b.territory),
                 "juveniles": sum(sorted(b.territory) == list(t) and isinstance(b, Juvenile) for b in m.type[Beaver] if b.territory),
                 "adults": sum(sorted(b.territory) == list(t) and isinstance(b, Adult) for b in m.type[Beaver] if b.territory), }
                for t in set(tuple(sorted(b.territory)) for b in m.type[Beaver] if b.territory)],

            "territory_size_cells": lambda m: [list(t) for t in set(tuple(sorted(b.territory)) for b in m.type[Beaver] if b.territory)],

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
