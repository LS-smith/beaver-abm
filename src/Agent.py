from mesa import Agent
import numpy as np
import random

class Beaver(Agent):
    """Base Beaver Class"""

    def __init__(self, model, sex=None, cell=None, age=0):
        """
		* Initialise and populate the model
		"""
        super().__init__(model) 
        self.sex = sex if sex else model.random.choice(['M', 'F'])
        self.partner = None
        self.age = age
        self.reproduction_timer = 0
        self.remove = False # mark for removal
        self.territory = set() # territory coords
        self.territory_abandonment_timer = None 
        self.dispersal_attempts = 0

    def step(self):
        #check territory timer
        if self.territory and self.territory_abandonment_timer is not None:
            self.territory_abandonment_timer -= 1
            if self.territory_abandonment_timer <= 0:
                self.abandon_territory()
        
        potential_mates = self.mate()
        if potential_mates:
            mate = self.random.choice(potential_mates)
            self.partner = mate
            mate.partner = self

        neighbours = self.model.grid.get_cell_list_contents([self.pos])
        if ( self.partner is None
            or getattr(self.partner, "remove", False)  # check if partner is not marked for removal
            or self.partner.partner != self
        ):
            # if no partner, or partner is marked for removal, or partner is not paired with self
            self.partner = None # clear partner
            potential_mates = [
                a for a in neighbours
                if ( isinstance(a, Beaver) 
                    and a.sex != self.sex and (a.partner is None or getattr(a.partner, "remove", False) 
                    or a.partner.partner !=a))]
            if potential_mates:
                mate = self.random.choice(potential_mates)
                self.partner = mate
                mate.partner = self

        #only move if doesnt have a territory - move together if paired else move alone.
        if not self.territory:
            if self.partner and self.partner.partner == self:
                if self.unique_id < self.partner.unique_id:  # only one of the pair moves both
                    self.colony()
            else:
                self.move()

    def mate(self, x=None, y=None, max_dist=None):
        mates=[]
        for a in self.model.type[Beaver]:
            if(a is not self
               and isinstance(a,Beaver)
               and a.sex!=self.sex
               and (a.partner is None or getattr(a.partner, "remove", False) or a.partner.partner != a)
               and a.territory
            ):
                if x is not None and y is not None and max_dist is not None:
                    tx, ty =np.mean([p[0] for p in a.territory]), np.mean([p[1] for p in a.territory])
                    dist = np.sqrt((tx - x) ** 2 + (ty - y) ** 2)
                    if dist > max_dist:
                        continue
                mates.append(a)
            return mates

    def disperse(self):
        mean_dispersal_distance = 1000 # 5km / 5m grid
        distance = int(np.random.exponential(mean_dispersal_distance))
        angle = np.random.uniform(0, 2 * np.pi)
        dx = int(distance *np.cos(angle))
        dy = int(distance *np.sin(angle))
        x0,y0 = self.pos
        x_new, y_new = np.clip(x0 + dx, 0, self.model.dem.shape[1]-1),  np.clip(y0 + dy, 0, self.model.dem.shape[0]-1)
        new_position = (x_new, y_new)

        potential_mates = self.mate(x_new,y_new, max_dist=500)
        if potential_mates:
            mate = self.random.choice(potential_mates)
            tx, ty =np.mean([p[0] for p in mate.territory]), np.mean([p[1] for p in mate.territory])
            self.model.grid.move_agent (self, (int(tx), int(ty)))
            self.partner =mate
            mate.partner = self
            self.dispersal_attempts = 0 
            print (f"Beaver {getattr(self, 'unique_id', id(self))} found mate at {(int(tx), int(ty))}")

        self.dispersal_attempts += 1
        if self.dispersal_attempts >= 4:
            print (f"Beaver {getattr(self, 'unique_id', id(self))} failed to disperse after 4 attempts. RIP ")
            self.remove = True
            return

    def move(self):
        possible_move = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        valid_move = []
        for pos in possible_move:
            x, y = pos
            if( 0<= x < self.model.dem.shape[1] and
                0<= y < self.model.dem.shape[0] and
                self.model.dem[y,x] != -100 ):
                valid_move.append(pos)
        if valid_move:
            new_area = self.random.choice(possible_move)
            self.model.grid.move_agent(self, new_area)
    
    def colony(self): #moving partner and kits together as a unit
        possible_move = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        valid_move = []
        for pos in possible_move:
            x, y = pos
            if( 0<= x < self.model.dem.shape[1] and
                0<= y < self.model.dem.shape[0] and
                self.model.dem[y,x] != -100):
                valid_move.append(pos)
        if valid_move:
            new_area = self.random.choice(possible_move)
            self.model.grid.move_agent(self, new_area)
            # move partner
            if self.partner and not getattr(self.partner, "remove", False):
                self.model.grid.move_agent(self.partner, new_area)
            # move kit
            for agent in self.model.grid.get_cell_list_contents([self.pos]):
                if isinstance(agent, Kit) and not getattr(agent, "remove", False): # check if partner is not marked for removal
                    self.model.grid.move_agent(agent, new_area)

    def form_territory(self):
        defend = set()
        for agent in self.model.type[Beaver]:
            if agent is not self and agent.territory:
                defend.update(agent.territory)
        # get 28 unoccupied cells around bevaer location
        x0, y0 =self.pos
        territory = set()
        for r in range(1, 10):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    x, y =x0 +dx, y0+dy
                    if (
                        0 <= x < self.model.dem.shape[1]
                        and 0 <= y < self.model.dem.shape[0]
                        and self.model.dem[y, x] != -100
                        and (x, y) not in defend ):
                        territory.add((x,y))
                    if len(territory) >= 28:
                        break
                if len(territory) >= 28:
                    break
            if len(territory) >= 28:
                break
            self.territory = territory
            self.territory_abandonment_timer = int(np.random.exponential(48))
            print(f"Beaver {getattr(self, 'unique_id', id(self))} formed territory at {self.pos} with {len(self.territory)} cells.")

    def abandon_territory(self):
        print(f"Beaver {getattr(self, 'unique_id', id(self))} abandoned territory at {self.pos} ")
        self.territory = set()
        self.territory_abandonment_timer = None
        # move partner and kits with agent
        if self.partner:
            self.partner.territory = set()
            self.partner.territory_abandonment_timer = None
            self.model.grid.move_agent(self.partner, self.pos)
        for agent in self.model.grid.get_cell_list_contents([self.pos]):
            if isinstance(agent, Kit):
                agent.territory = set()
                agent.territory_abandonment_timer = None
                self.model.grid.move_agent(agent, self.pos)

    def reproduce(self):
        if self.partner is not None:
            for _ in range(self.random.randint(1, 3)): # random number of kits between 1-3
                kit = Kit(self.model, sex=self.sex)
                self.model.grid.place_agent(kit, self.pos)
                self.model.type[Beaver].append(kit)

    def age_up(self):
        # kit -> juvenile at age 2 (24 steps), juvenile -> adult at age 3 (36 steps)
        if isinstance(self, Kit) and self.age >= 24: 
            return Juvenile(self.model, sex=self.sex, cell=self.cell, age=self.age)
        elif isinstance(self, Juvenile) and self.age >= 36:
            return Adult(self.model, sex=self.sex, cell=self.cell, age=self.age)
        else:
            return self


class Kit(Beaver):
    # kits move with group, can't pair or reproduce, age up
         #TODO: finish later, should only move with parents or die - think this will mess up when parent dead so add in that 

    def step(self):
        self.age += 1  

        neighbours = self.model.grid.get_cell_list_contents([self.pos]) # move with colony
        adults = [a for a in neighbours if isinstance(a, Adult)] #find adulgt in same cell
        if not adults:
            self.remove = True
            return

        new_self = self.age_up() # age up if applicable
        if new_self is not self:
            self.remove = True
            self.model.grid.place_agent(new_self, self.pos)
            self.model.type[Beaver].append(new_self)
            # return new_self.step()
            return


class Juvenile(Beaver):
    # juveniles disperse away from group, pair and reproduce, !build dams!, age up

    def step(self):
        self.age += 1  

        #assign territory
        if not self.territory:
            self.form_territory()
            print(f"Beaver {getattr(self, 'unique_id', id(self))} formed territory at {self.pos} with {len(self.territory)} cells.")

        # reproduction logic 
        if self.partner and self.partner.partner == self and self.unique_id < self.partner.unique_id:
            self.reproduction_timer += 1
            if self.reproduction_timer >= 12:
                self.reproduce()
                self.reproduction_timer = 0
        else:
            self.reproduction_timer = 0

        new_self = self.age_up() # age up if applicable
        if new_self is not self:
            self.remove = True
            self.model.grid.place_agent(new_self, self.pos)
            self.model.type[Beaver].append(new_self)
            # return new_self.step() - no need to call step again, mutating the agent list by iterating
            return

class Adult(Beaver):
    # adults have full range of beaver behaviour (pairing, moving, reproducing, !building dams!, they dont age up-they die)
    def step(self):
        self.age += 1
        super().step()  # call base beaver logic (pairing, movement)

        # reproduction logic 
        if self.partner and self.partner.partner == self and self.unique_id < self.partner.unique_id:
            self.reproduction_timer += 1
            if self.reproduction_timer >= 12:
                self.reproduce()
                self.reproduction_timer = 0
        else:
            self.reproduction_timer = 0

        #TODO: partners dont re-pair when partner dies - they also dont move! fix
        if self.age >= 84: 
            # break pair bond if partner is alive
            if self.partner and self.partner.partner == self:
                self.partner.partner = None
            self.partner = None # clear self.partner
            self.remove = True
            return
