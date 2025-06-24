from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.experimental.devs import ABMSimulator
import numpy as np
from rasterio import open as rio_open

from Agent import Beaver, Kit, Juvenile, Adult # if this is seperate files

class BeaverModel(Model):
    def __init__(self, dem, initial_beavers=50, seed=None, simulator=None): # initialise
        super().__init__(seed=seed)

        
        self.dem = dem
        self.height, self.width = self.dem.shape

        # properly initialise the grid
        self.grid = OrthogonalVonNeumannGrid(
            [self.height, self.width],
            torus=True,
            capacity=float("inf"),
            random=self.random,
        )

        # initialise type as a set NOT list
        self.type = {Beaver: []}

        # create initial beavers and add them to the grid
        for _ in range(initial_beavers):
            cell = self.random.choice(self.grid.all_cells.cells)
            beaver = Adult(model=self, cell=cell) # add only adult beavers 
            cell.agents.append(beaver)
            self.type[Beaver].append(beaver)

        print("aftermodel creation:")
        print("beavers in model.type[Beaver]:", len(self.type[Beaver]))
        print("total number of agents in the grid:", sum(len(cell.agents) for cell in self.grid.all_cells.cells))


        self.datacollector = DataCollector({
            "Beavers": lambda m: len(m.type[Beaver]),
            "Paired Beavers": lambda m: len(
                [a for a in m.type[Beaver] if a.partner and a.unique_id < a.partner.unique_id]
            ),
            "Males": lambda m: len([a for a in m.type[Beaver] if a.sex == "M"]),
            "Females": lambda m: len([a for a in m.type[Beaver] if a.sex == "F"]),
            "Kits": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Kit)]),
            "Juveniles": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Juvenile)]),
            "Adults": lambda m: len([a for a in m.type[Beaver] if isinstance(a, Adult)]),
        })
        self.datacollector.collect(self)

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
                if agent in agent.cell.agents:
                    agent.cell.agents.remove(agent)
                if agent in self.type[Beaver]:
                    self.type[Beaver].remove(agent)
        
        self.datacollector.collect(self) # collect data on each step


