
import numpy as np
import geopandas as gdp
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.experimental.devs import ABMSimulator
from mesa.space import MultiGrid
from rasterio import open as rio_open

from Agent import Beaver, Kit, Juvenile, Adult, Dam


class Flood_Model(Model):
    #ABM model for beaver population and dam building simulation
    
    def __init__(self, dem, dem_transform, initial_beavers=50, seed=None, simulator=None):
        super().__init__(seed=seed)

        # Load habitat suitability model
        with rio_open('./data/hsm_5m.tif') as hsm:
            self.hsm = hsm.read(1)

        # Load distance to water layer
        with rio_open('./data/distance_to_water_5m.tif') as dtw:
            self.distance_to_water = dtw.read(1)

        # Load waterway network
        self.waterways = gdp.read_file('./data/Water_network.shp')

        self.dem = dem
        self.dem_transform = dem_transform
        self.height, self.width = self.dem.shape

        # Initialise the spatial grid
        self.grid = MultiGrid(self.width, self.height, torus=True)
        
        # Initialise agent type collections
        self.type = {Beaver: [], Dam: []}

        # Create initial beaver pairs
        num_pairs = initial_beavers // 2

        # Find suitable cells near water for initial placement
        suitable_cells = np.argwhere(
            (self.hsm >= 2) & (self.hsm <= 4) & (self.distance_to_water < 50))

        used_indices = set()
        pairs_placed = 0

        while pairs_placed < num_pairs:
            idx1 = self.random.choice(range(len(suitable_cells)))
            if idx1 in used_indices:
                continue
            y1, x1 = suitable_cells[idx1]
            
            # Find an adjacent suitable cell not already used
            neighbours = set(
                (y1 + dy, x1 + dx)
                for dy in [-1, 0, 1]
                for dx in [-1, 0, 1]
                if not (dy == 0 and dx == 0)
            )
            neighbour_indices = [
                i for i, (y, x) in enumerate(suitable_cells)
                if (y, x) in neighbours and i not in used_indices
            ]
            
            if neighbour_indices:
                idx2 = self.random.choice(neighbour_indices)
                y2, x2 = suitable_cells[idx2]
                
                # Create mated pair
                male = Adult(self, sex="M")
                female = Adult(self, sex="F")
                male.partner = female
                female.partner = male
                female.breeding_month = self.random.choice([4, 5, 6])
                female.kits_this_year = False
                
                # Place agents on grid
                self.grid.place_agent(male, (x1, y1))
                self.grid.place_agent(female, (x2, y2))
                self.type[Beaver].append(male)
                self.type[Beaver].append(female)
                
                used_indices.update([idx1, idx2])
                pairs_placed += 1
            else:
                used_indices.add(idx1)

        # Initialise data collection
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

            #Territory location as centroid
            "territory_location": lambda m: [(sum(xs)/len(t), sum(ys)/len(t)) if len(t) > 0 else (None, None) for t in set(tuple(b.territory) for b in m.type[Beaver] if b.territory) for xs, ys in [zip(*t)] if len(t) > 0],

            "territory_cells_location": lambda m: [list (t) for t in set(tuple(b.territory) for b in m.type[Beaver] if b.territory) ],

            "dam_num": lambda m: len(m.type[Dam]),

            "dam_locations": lambda m: [d.pos for d in m.type[Dam]],

            "flooded_cells": lambda m: sum(np.sum(d.flooded_area) if hasattr(d, "flooded_area") and d.flooded_area is not None else 0 for d in m.type[Dam]),

            "flood_cell_location": lambda m: [[(r, c) for r, c in zip(*np.where(d.flooded_area == 1))] if hasattr(d, "flooded_area") and d.flooded_area is not None else [] for d in m.type[Dam]],

        })

        self.datacollector.collect(self)

        # Set up simulator
        if simulator is not None:
            self.simulator = simulator
            self.simulator.setup(self)
        self.running = True
        self.month = 1

    def step(self):
        # Advance month counter
        self.month += 1
        if self.month > 12:
            self.month = 1

        # Update all beaver agents
        for agent in list(self.type[Beaver]):
            agent.step()

        # Update all dam agents
        for dam in list(self.type[Dam]):
            dam.step()

        # Remove agents marked for removal
        beavers_to_remove = [agent for agent in self.type[Beaver] 
                           if getattr(agent, "remove", False)]
        for agent in beavers_to_remove:
            if agent.pos is not None:
                try:
                    self.grid.remove_agent(agent)
                except ValueError:
                    pass   
            self.type[Beaver].remove(agent)

        # Remove dams marked for removal
        dams_to_remove = [dam for dam in self.type[Dam] if dam.dam_remove]
        for dam in dams_to_remove:
            if dam.pos is not None:
                try:
                    self.grid.remove_agent(dam)
                except ValueError:
                    pass
            self.type[Dam].remove(dam)
        
        # Collect data for this step
        self.datacollector.collect(self)
