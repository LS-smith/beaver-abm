from mesa import Agent
import numpy as np
import random
from scipy.ndimage import label
from numpy import zeros

class Beaver(Agent):
    """Base Beaver Class"""

    def __init__(self, model, sex=None, age=0):
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
                self.model.dem[y,x] != 0 ):
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
                self.model.dem[y,x] != 0):
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
        
        mean = np.log(3000) #mean bankful length of terriroty is ~3km CHECK!!!
        sigma = 1.0
        bank_length = np.random.lognormal(mean=mean, sigma=sigma)
        bank_length = np.clip(bank_length, 500, 30000) #spread is 0.5 - 3km CHECK!!!!

        cell_length = getattr(self.model.grid, "cell_width", 5)
        territory_cells = int(bank_length / cell_length)
        territory_cells = max(territory_cells, 1)

        occupied = set()
        for agent in self.model.type[Beaver]:
            if agent is not self and agent.territory:
                occupied.update(agent.territory)

        hsm = self.model.hsm
        distance_to_water = self.model.distance_to_water
        water_mask = (hsm == 5)

        max_bank_dist = 20 #100m - check!!!!
        fuzzy_water_dist = np.exp(-distance_to_water / max_bank_dist)
        habitat_mask = (hsm >= 2) & (hsm < 5)
        fuzzy_mask = habitat_mask * fuzzy_water_dist

        for x, y in occupied:
            if 0 <= x < fuzzy_mask.shape[1] and 0 <= y < fuzzy_mask.shape[0]:
                fuzzy_mask[y, x] = 0

        structure = np.array([[0,1,0],[1,1,1],[0,1,0]]) # rooks move connectivity - patches together are connected
        labelled, patch_count = label(fuzzy_mask, structure = structure)

        best_patch = None
        best_score = -np.inf
        for patch_id in range(1, patch_count+1):
            patch_cells = np.argwhere(labelled == patch_id)
            patch_size = patch_cells.shape[0]
            if patch_size < territory_cells // 20: # skips patches that are very small 1/20th of bankful length, ask Jonny! may need to tweak more to allow for increased fragmentation nin high competition areas.
                continue
            
            patch_hsms = hsm[tuple(patch_cells.T)]
            patch_fuzzy = fuzzy_mask[tuple(patch_cells.T)]
            mean_hsm = np.mean(patch_hsms[patch_hsms < 5])
            mean_fuzzy = np.mean(patch_fuzzy)

            scale = np.interp(mean_hsm, [2,4], [0.5, 2.0])
            territory_cells_scaled = int(territory_cells * scale)
            territory_cells_scaled = max(territory_cells_scaled, 1)


            score = mean_hsm * 100 + mean_fuzzy * 100 -abs(patch_size - territory_cells_scaled)
            if any((self.pos[1], self.pos[0]) == tuple(cell) for cell in patch_cells):
                score += 1000 # make likely to choose patch in current position
            if score > best_score:
                best_score = score
                best_patch = patch_cells

        if best_patch is not None:
            if best_patch.shape[0] > territory_cells_scaled:
                chosen_territory = np.random.choice(best_patch.shape[0], territory_cells_scaled, replace=False)
                best_patch = best_patch[chosen_territory]
            self.territory = set((int(cell[1]), int(cell[0])) for cell in best_patch)
            self.territory_abandonment_timer = int(np.random.exponential(48))
            print(f"Beaver {getattr(self, 'unique_id', id(self))} formed territory at {self.pos} with {len(self.territory)} cells (length: {bank_length:0f}m).")
        else:
            self.territory = set()
            print(f"Beaver {getattr(self, 'unique_id', id(self))} could not find territory. Poor beaver.")



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
            return Juvenile(self.model, sex=self.sex, age=self.age)
        elif isinstance(self, Juvenile) and self.age >= 36:
            return Adult(self.model, sex=self.sex, age=self.age)
        else:
            return self



    def build_dam(self):
        if not self.territory: #if not in territory
            return

        territory_shape = self.model.grid.cells_to_geometry(self.territory) #get territory shape

        water_in_territory = self.model.waterways[self.model.waterways.intersects(territory_shape)] #find all waterways intersecting territory
        if water_in_territory.empty:
            return

        for idx, segment in water_in_territory.iterrows(): 
            gradient = segment["gradient"]
            if gradient == 'NULL' or (isinstance(gradient, float) and np.isnan(gradient)):
                gradient = 0
                
            if segment["gradient"] > 30: #only build dam if gradient lower than 3%
                continue

            channel = segment.geometry #all points along the channel
            num_points = int(channel.length //5) #every 5m can build
            for fraction in np.linspace(0,1, num_points):
                point = channel.interpolate(fraction * channel.length)
                x,y = int(point.x), int(point.y)
                if (x,y) not in self.territory:
                    continue
            
            temp_dam = Dam(self.model, (x, y), depth=depth)
            flood_layer = temp_dam.flood_fill()
            if temp_dam.floods_non_water():
                self.model.grid.place_agent(temp_dam, (x, y))
                self.dam = temp_dam
                print(f"Beaver {getattr(self, 'unique_id', id(self))} built dam at {(x, y)}")
                return
        
        print ("Dam not built: too much water man!")

class Kit(Beaver):
    # kits move with group, can't pair or reproduce, age up
    def __init__(self, model, sex=None, age=0):
        super().__init__(model, sex=sex, age=age)

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
            return



class Juvenile(Beaver):
    # juveniles disperse away from group, pair and reproduce, !build dams!, age up
    def __init__(self, model, sex=None, age=0):
        super().__init__(model, sex=sex, age=age)

    def step(self):
        self.age += 1  
        super().step()

        #assign territory
        if not self.territory:
            self.disperse()
            if self.remove:
                return
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
    def __init__(self, model, sex=None, age=0):
        super().__init__(model, sex=sex, age=age)
    
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

class Dam(Agent):
    def __init__(self, model, pos, depth):
        super().__init__(model)
        self.pos = pos
        self.depth = depth
        self.flooded_area = None

        #depth = varied

        def flood_fill(self):
            x0, y0 = self.pos
            dem = self.model.dem
            flood_layer = zeros(dem.shape)
            r0, c0 = self.model.grid.cells_to_index(x0, y0)
            assessed = set()
            to_assess = set()
            to_assess.add((r0, c0))
            flood_extent = dem[r0, c0] + self.depth

            while to_assess:
                r, c = to_assess.pop()
                assessed.add((r, c))
                if dem[r, c] <= flood_extent:
                    flood_layer[r, c] = 1
                    for r_adj, c_adj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        neighbour = (r + r_adj, c + c_adj)
                        if 0 <= neighbour[0] < dem.shape[0] and 0 <= neighbour[1] < dem.shape[1] and neighbour not in assessed:
                            to_assess.add(neighbour)
            self.flooded_area = flood_layer
            return flood_layer

        def flood_land(self):
            hsm = self.model.hsm
            flooded_indices = np.argwhere(self.flooded_area == 1)
            for r, c in flooded_indices:
                if hsm[r, c] != 6:
                    return True
            return False
    