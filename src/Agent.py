from mesa import Agent
import numpy as np
import random
from scipy.ndimage import label
from numpy import zeros
import time
from shapely.geometry import MultiPoint
import rasterio

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
        self.remove = False # mark for removal
        self.territory = set() # territory coords
        self.territory_abandonment_timer = None 
        self.dispersal_attempts = 0
        self.death_age = min(int(np.random.exponential(84)), 240)
        self.kits_this_year = False
        self.breeding_month = None


    def step(self):

        #print(f"Agent {getattr(self, 'unique_id', id(self))} starting step")

        # form territory if not already done
        if not self.territory:
            if self.partner and self.partner.partner == self:
                if self.unique_id < self.partner.unique_id:  # Only one of the pair forms territory
                    self.form_territory()
                    self.partner.territory = set(self.territory)
            else:
                self.form_territory()
            
        
        #check territory timer
        if self.territory and self.territory_abandonment_timer is not None:
            self.territory_abandonment_timer -= 1
            if self.territory_abandonment_timer <= 0:
                self.abandon_territory()
        
        if ( self.partner is None
            or getattr(self.partner, "remove", False)  # check if partner is not marked for removal
            or self.partner.partner != self
        ):
            # if no partner, or partner is marked for removal, or partner is not paired with self
            self.partner = None # clear partner

            potential_mates = self.mate(self.pos[0], self.pos[1], max_dist = 1000)
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
                self.disperse()
                if self.remove:
                    return
                
    
        if self.territory:
            self.build_dam()

        for dam in self.model.type[Dam]:
            if dam.pos in self.territory and dam.abandoned and dam.repairable:
                dam.abandoned = False
                dam.decay_timer = None
                dam.repairable = False
                #print(f"Dam at {dam.pos} repaired by beaver {getattr(self, 'unique_id', id(self))}")

        if np.random.rand() < 0.005:
            #print(f"Beaver {getattr(self, 'unique_id', id(self))} died. rip.")
            self.remove = True
            return


        #reproduction timer
        if self.model.month ==1:
            self.kits_this_year = False
            if self.sex == "F":
                self.breeding_month = self.random.choice([4, 5, 6])

        # reproduction logic 
        if (self.sex =="F" and
            self.partner and self.partner.partner == self 
            and self.model.month == self.breeding_month and  #april, may or june
            not self.kits_this_year):

            self.reproduce()
            self.kits_this_year = True

        self.age += 1
        if self.age >= self.death_age:
            self.remove = True
            return

    def mate(self, x=None, y=None, max_dist=1000):
        mates=[]
        for a in self.model.type[Beaver]:
            if(a is not self
               and isinstance(a,Beaver)
               and a.sex!=self.sex
               and (a.partner is None or getattr(a.partner, "remove", False) or a.partner.partner != a)
            ):
                if x is not None and y is not None and max_dist is not None:
                    if a.territory:
                        tx, ty =np.mean([p[0] for p in a.territory]), np.mean([p[1] for p in a.territory])
                    else:
                        tx, ty = a.pos
                    dist = np.sqrt((tx - x) ** 2 + (ty - y) ** 2)
                    if dist > max_dist:
                        continue
                mates.append(a)
            return mates
        
    def disperse(self):
        mean_dispersal_distance = 1000 # 5km / 5m grid
        cell_width = getattr(self.model.grid, "cell_width", 5)
        water_threshold = 100

        distance = int(np.random.exponential(mean_dispersal_distance))
        angle = np.random.uniform(0, 2 * np.pi)
        dx = int(distance *np.cos(angle))
        dy = int(distance *np.sin(angle))
        x0,y0 = self.pos
        x_new, y_new = np.clip(x0 + dx, 0, self.model.dem.shape[1]-1),  np.clip(y0 + dy, 0, self.model.dem.shape[0]-1)
        
        potential_mates = self.mate(x_new,y_new, max_dist=distance)
        if potential_mates:
            mate = self.random.choice(potential_mates)
            tx, ty =np.mean([p[0] for p in mate.territory]), np.mean([p[1] for p in mate.territory])
            self.model.grid.move_agent (self, (int(tx), int(ty)))
            self.partner =mate
            mate.partner = self
            self.dispersal_attempts = 0 
            print (f"Beaver {getattr(self, 'unique_id', id(self))} found mate at {(int(tx), int(ty))}")
            return

        distance_to_water = self.model.distance_to_water
        possible_cells = np.argwhere(distance_to_water <= water_threshold)
        if possible_cells.size > 0:
            distances = np.linalg.norm(possible_cells - np.array([y_new, x_new]), axis = 1)
            best_idx = np.argmin(distances)
            y_final, x_final = possible_cells[best_idx]
        else:
            x_final, y_final = x_new, y_new

        self.model.grid.move_agent(self, (int(x_final), int(y_final)))

        old_pos = self.pos
        self.form_territory()
        if self.territory:
            self.dispersal_attempts = 0
            #print (f"Beaver {getattr (self, 'unique_id', id(self))} formed new territory at {(x_final, y_final)}")
            return
        else:
            self.model.grid.move_agent(self, old_pos)
            self.dispersal_attempts += 1
            #print (f"Beaver {getattr (self, 'unique_id', id(self))} found no suitable territory at {(x_final, y_final)}, dispersal attempt {self.dispersal_attempts} / 5 ")

        if self.dispersal_attempts >= 5:
            #print (f"Beaver {getattr(self, 'unique_id', id(self))} failed to disperse after 5 attempts. It is winter now, and without provisions they will surely perish. RIP ")
            self.remove = True

        if (self.partner is None
            or getattr(self.partner, "remove", False)
            or self.partner.partner != self
        ):
            potential_mates = self.mate(self.pos[0], self.pos[1], max_dist=1000)
            if potential_mates:
                mate = self.random.choice(potential_mates)
                self.partner = mate
                mate.partner = self

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
            new_area = self.random.choice(valid_move)
            if self.pos != new_area:
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
            new_area = self.random.choice(valid_move)
            if self.pos != new_area:
                self.model.grid.move_agent(self, new_area)
            # move partner
            if self.partner and not getattr(self.partner, "remove", False):
                if self.partner.pos != new_area:
                    self.model.grid.move_agent(self.partner, new_area)
            # move kit
        for agent in self.model.grid.get_cell_list_contents([self.pos]):
            if isinstance(agent, Kit) and not getattr(agent, "remove", False):
                if agent.pos != new_area:
                    self.model.grid.move_agent(agent, new_area)



    def form_territory(self):
        cell_length = getattr(self.model.grid, "cell_width", 5)
        mean = np.log(3000) #mean bankful length of terriroty is ~3km CHECK!!!
        sigma = 0.5
        bank_length = np.random.lognormal(mean=mean, sigma=sigma)
        bank_length = np.clip(bank_length, 500, 20000) #spread is 0.5 - 20 km CHECK!!!! that would be 10 and 600 squares you idiot
        territory_cells = max(int(bank_length / cell_length), 1)

        occupied = set()
        for agent in self.model.type[Beaver]:
            if agent is not self and agent.territory:
                occupied.update(agent.territory)

        hsm = self.model.hsm
        distance_to_water = self.model.distance_to_water
        #water_mask = (hsm == 5)

        waterway_buffer = 100
        mask =(distance_to_water <= waterway_buffer) & (hsm >=2) & (hsm <=4)
        if occupied:
            occupied_array = np.array(list(occupied))
            mask[occupied_array[:,1], occupied_array[:,0]] = 0

        
        radius = 600 #3km
        cx, cy = self.pos
        y_indices, x_indices = np.indices(mask.shape)
        radius_mask = ((x_indices - cx) ** 2 + (y_indices - cy) ** 2) <= radius ** 2
        mask = mask & radius_mask
        
        mask[cy, cx] = True
        
        labeled_array, num_features = label(mask)
        territory_label = labeled_array[cy, cx]

        if territory_label == 0:
        # No connected region at beaver's location
            self.territory = set()
            #print(f"Beaver {getattr(self, 'unique_id', id(self))} could not form territory at {self.pos}.")
            return

        visited = set()
        patch = []
    
        def cell_score(y, x):
            if hsm[y, x] in [2, 3, 4]:
                return -int(hsm[y, x]) + float(distance_to_water[y, x]) * 0.01
            else:
                return np.inf 

        frontier = [(cell_score(cy, cx), cy, cx)]
        while frontier and len(patch) < territory_cells:
            frontier.sort()
            score, y, x = frontier.pop(0)
            if (y, x) in visited:
                continue
            visited.add((y, x))
        
            if labeled_array[y, x] == territory_label and hsm[y, x] in [2, 3, 4]:# only add if not unsuitable or water
                patch.append((x, y))
            
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]: #rooks move
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < labeled_array.shape[0] and
                        0 <= nx < labeled_array.shape[1] and
                        (ny, nx) not in visited and
                        labeled_array[ny, nx] == territory_label and
                        hsm[ny, nx] in [2, 3, 4]):
                        frontier.append((cell_score(ny, nx), ny, nx))

        self.territory = set(patch)
        self.territory_abandonment_timer = int(np.random.exponential(48)) # most territories last 4 years some much longer
        #print(f"Beaver {getattr(self, 'unique_id', id(self))} formed contiguous territory at {self.pos} with {len(self.territory)} cells.")

        # assign the same territory to partner
        if self.partner and self.partner.partner == self:
            self.partner.territory = self.territory
            self.partner.territory_abandonment_timer = self.territory_abandonment_timer

        # assign the same territory to all offspring
        for agent in self.model.grid.get_cell_list_contents([self.pos]):
            if isinstance(agent, (Kit, Juvenile)) and not getattr(agent, "remove", False):
                agent.territory = self.territory
                agent.territory_abandonment_timer = self.territory_abandonment_timer

    def abandon_territory(self):
        #print(f"Beaver {getattr(self, 'unique_id', id(self))} abandoned territory at {self.pos} ")
        self.territory = set()
        self.territory_abandonment_timer = None
        # move partner and kits with agent

        for dam in self.model.type[Dam]:
            if dam.pos in self.territory and not dam.abandoned:
                dam.abandoned = True
                dam.decay_timer = int(np.random.normal(24, 6))  # ~2 years, kernel spread
                dam.repairable = True
                #print(f"Dam at {dam.pos} abandoned. Time to decay is {dam.decay_timer}")


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
            for _ in range(self.random.randint(1, 4)): # random number of kits between 1-4
                kit = Kit(self.model, sex=self.random.choice(['M', 'F']))
                kit.territory = self.territory  # inherit parent's territory
                kit.territory_abandonment_timer = self.territory_abandonment_timer
                self.model.grid.place_agent(kit, self.pos)
                self.model.type[Beaver].append(kit)
                #print(f"Kit created at {self.pos} by parent {getattr(self, 'unique_id', id(self))}"){self.pos} by parent {getattr(self, 'unique_id', id(self))}")


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
            #print("No territory, skipping dam build.")
            return

        if self.territory:
            for agent in self.model.type[Beaver]:
                if(agent is not self
                   and agent.territory == self.territory
                   and agent.unique_id<self.unique_id):
                    return

        transform = self.model.dem_transform
        inv_transform = ~transform
        real_coords = [rasterio.transform.xy(transform, y, x, offset='center') for (x, y) in self.territory]
        territory_shape = MultiPoint(real_coords).convex_hull #get territory shape
        #print(f"Territory bounds: {territory_shape.bounds}")

        water_in_territory = self.model.waterways[
            (self.model.waterways.intersects(territory_shape)) & #find all waterways intersecting territory
            (self.model.waterways["gradient"] <= 4)] #grad >4
        #print(f"Waterways intersecting territory: {len(water_in_territory)}")
        
        if water_in_territory.empty:
            #print("No waterways intersecting territory.")
            return
        dam_attempts = 0
        for idx, segment in water_in_territory.iterrows(): 
            if dam_attempts >= 5:
                #print("Max dam attempts reached for this step.")
                break
            gradient = segment["gradient"]
            #print(f"Checking segment {idx} with gradient {gradient}")
            if gradient == 'NULL' or (isinstance(gradient, float) and np.isnan(gradient)):
                gradient = 0

            # if segment["gradient"] > 4: #only build dam if gradient lower than 4%
            #   print(f"Segment {idx} gradient too high ({segment['gradient']}), skipping.")
            #  continue

            channel = segment.geometry #all points along the channel
            num_points = int(channel.length //5) #every 5m can build
            for fraction in np.linspace(0,1, num_points):
                if dam_attempts >= 5:
                    break
                point = channel.interpolate(fraction * channel.length)
                col, row = inv_transform * (point.x, point.y)
                x,y = int(round(col)), int(round(row))
                if (x,y) not in self.territory:
                    continue

                existing_dams = self.model.grid.get_cell_list_contents([(x,y)])
                if any(isinstance(a, Dam) for a in existing_dams):
                    #print(f"Dam already exists at {(x, y)}, skipping.")
                    continue
            
                temp_dam = Dam(self.model, (x, y), depth = None)
                flood_layer = temp_dam.flood_fill()
                if temp_dam.flood_land():
                    if temp_dam.pos != (x,y):
                        self.model.grid.place_agent(temp_dam, (x, y))
                    self.model.type[Dam].append(temp_dam)
                    self.dam = temp_dam
                    #print(f"Beaver {getattr(self, 'unique_id', id(self))} built dam at {(x, y)}")
                    flooded_indices = np.argwhere(temp_dam.flooded_area == 1)
                    for r, c in flooded_indices:
                        if 0 <= r < self.model.hsm.shape[0] and 0 <= c < self.model.hsm.shape[1]:
                            self.model.hsm[r, c] = 6 
                    return
                else:
                    #print ("Dam not built: too much water man!")
                    dam_attempts +=1
    


class Kit(Beaver):
    # kits move with group, can't pair or reproduce, age up
    def __init__(self, model, sex=None, age=0):
        super().__init__(model, sex=sex, age=age)

    def step(self):
        
        if np.random.rand() < 0.005:
            #print(f"Kit {getattr(self, 'unique_id', id(self))} died due to background mortality.")
            self.remove = True
            return

        self.age += 1
        if hasattr(self, "death_age") and self.age >= self.death_age:
            self.remove = True
            return

        if self.territory:
            adults_in_territory = [a for a in self.model.type[Beaver] if isinstance(a, Adult) and a.territory and self.territory and a.territory == self.territory]
            if not adults_in_territory:
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
        self.helper_timer = np.random.randint(12, 36)

    def step(self):
        super().step()
        if self.remove:
            return
        if self.helper_timer >0:
            self.helper_timer -= 1
            return

        

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
        super().step()  # call base beaver logic (pairing, movement)
        if self.remove:
            return
        
        

        

class Dam(Agent):
    def __init__(self, model, pos, depth):
        super().__init__(model)

        self.pos = pos

        self.abandoned = False
        self.decay_timer =None
        self.repairable = False

        if depth is None:
            mu, sigma = 1.6, 0.44 #hartman 2006
            lower, upper = 0.55, 2.0
            while True:
                d = np.random.normal(mu, sigma)
                if lower <= d <= upper:
                    self.depth = d
                    break
        else:
            self.depth = depth
        self.flooded_area = None

    def flood_fill(self):
        x0, y0 = self.pos
        dem = self.model.dem
        flood_layer = zeros(dem.shape, dtype = int)
        r0, c0 = y0, x0
        assessed = set()
        to_assess = set()
        to_assess.add((r0, c0))
        flood_extent = dem[r0, c0] + self.depth

        while to_assess:
            r, c = to_assess.pop()
            assessed.add((r, c))
            if dem[r, c] <= flood_extent:
                flood_layer[r, c] = 1
                for r_adj, c_adj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:                        neighbour = (r + r_adj, c + c_adj)
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
    
    def step(self):
        if self.abandoned:
            if self.decay_timer is not None:
                self.decay_timer -= 1
                if self.decay_timer <= 0:
                    #print(f"Dam at {self.pos} decayed and removed.")
                    flooded_indices = np.argwhere(self.flooded_area == 1)
                    for r, c in flooded_indices:
                        if 0 <= r < self.model.hsm.shape[0] and 0 <= c < self.model.hsm.shape[1]:
                            self.model.hsm[r, c] = 0  # Reset flooded cells
                    self.remove = True
                    self.repairable = False
    