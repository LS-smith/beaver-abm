from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.experimental.devs import ABMSimulator
import numpy as np
from rasterio import open as rio_open

from Agent import Beaver, Kit, Juvenile, Adult, Dam # if this is seperate files

class Flood_Model(Model):
    def __init__(self, dem, initial_beavers=50, seed=None, simulator=None): # initialise
        super().__init__(seed=seed)

        with rio_open('/Users/r34093ls/Documents/GitHub/beaver-abm/data/hsm_5m.tif') as hsm:
            self.hsm = hsm.read(1)

        with rio_open('/Users/r34093ls/Documents/GitHub/beaver-abm/data/distance_to_water_5m.tif') as dtw:
            self.distance_to_water = dtw.read(1)

        
        self.dem = dem
        self.height, self.width = self.dem.shape

        # properly initialise the grid
        self.grid = MultiGrid(self.width, self.height, torus=True)

        # initialise type as a set NOT list
        self.type = {Beaver: [], Dam: []}

        valid_area =[(x,y)
                    for y in range(self.height)
                    for x in range(self.width)
                    if self.dem[y,x] != 0]

        # create initial beavers and add them to the grid
        for _ in range(initial_beavers):
            x, y =self.random.choice(valid_area)
            #x = self.random.randrange(self.width)
            #y = self.random.randrange(self.height)
            beaver = Juvenile(self) # add only adult beavers (may be self.unique_id)
            self.grid.place_agent(beaver, (x,y))
            self.type[Beaver].append(beaver)

        print("aftermodel creation:")
        print("beavers in model.type[Beaver]:", len(self.type[Beaver]))
        print("total number of agents in the grid:", sum(len(cell_contents) for cell_contents, pos in self.grid.coord_iter()))


        self.datacollector = DataCollector({
            "Beaver_Count": lambda m: len(m.type[Beaver]),
            "Paired Beavers": lambda m: len([a for a in m.type[Beaver] if a.partner and a.unique_id < a.partner.unique_id]),
            "Males": lambda m: len([a for a in m.type[Beaver] if a.sex == "M"]),
            "Females": lambda m: len([a for a in m.type[Beaver] if a.sex == "F"]),
            "Kits": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Kit)]),
            "Juveniles": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Juvenile)]),
            "Adults": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Adult)]),
            "Dams": lambda m: len(m.type[Dam])
        })
        self.datacollector.collect(self)

def stats (model, step):
     agents = model.type[Beaver]
     dams = model.type[Dam]
     pop_size = len(agents)
     num_kits= sum(isinstance(a, Kit) for a in agents)
     num_juveniles = sum(isinstance(a, Juvenile) for a in agents)
     num_adults = sum(isinstance(a, Adult) for a in agents)
     num_dam = len(dams)
     flooded_cells = sum(np.sum(dam.flooded_area) for dam in dams if dam.flooded_area is not None)


        if simulator is not None:
            self.simulator = simulator
            self.simulator.setup(self)   
        self.running = True

    def step(self):
        # update the agents
        for agent in list(self.type[Beaver]):
            agent.step()

        for agent in list(self.type[Beaver]):
            if getattr(agent, "remove", False):
                self.grid.remove_agent(agent)
                self.type[Beaver].remove(agent)
        
        self.datacollector.collect(self) # collect data on each step


