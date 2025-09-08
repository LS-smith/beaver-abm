
import random
import time
import numpy as np
import rasterio
from mesa import Agent
from numpy import zeros
from scipy.ndimage import label
from shapely.geometry import MultiPoint


class Beaver(Agent):
    # Base Beaver class with mating, territory formation, dispersal, and dam building
    
    def __init__(self, model, sex=None, age=0):
        super().__init__(model)
        self.sex = sex if sex else model.random.choice(['M', 'F'])
        self.partner = None
        self.age = age
        self.remove = False  # Mark for removal
        self.territory = set()  # Territory coordinates
        self.territory_abandonment_timer = None
        self.dispersal_attempts = 0
        self.death_age = min(int(np.random.exponential(84)), 240)
        self.kits_this_year = False
        self.breeding_month = None

    def step(self):
        # Execute one time step of beaver behaviour
        # Check for territory and mate availability
        if not self.territory and (self.partner is None
                                   or getattr(self.partner, "remove", False)
                                   or self.partner.partner != self):
            if self.dispersal_attempts < 5:
                mate_search_radius = 4000  # 20km
                mate = self.mate(self.pos[0], self.pos[1], max_dist=mate_search_radius)
                if mate:
                    pass
                else:
                    self.dispersal_attempts += 1
                    self.disperse()
                    return
            else:
                self.form_territory()
                if not self.territory:
                    self.remove = True
                    return
                self.dispersal_attempts = 0

        # Form territory if not already done
        if not self.territory:
            if self.partner and self.partner.partner == self:
                # Only one of the pair forms territory
                if self.unique_id < self.partner.unique_id:
                    self.form_territory()
                    self.partner.territory = set(self.territory)
            else:
                self.form_territory()

        # Check territory timer
        if self.territory and self.territory_abandonment_timer is not None:
            self.territory_abandonment_timer -= 1
            if self.territory_abandonment_timer <= 0:
                self.abandon_territory()

        # Only move if doesn't have territory - move together if paired
        if not self.territory:
            if self.partner and self.partner.partner == self:
                # Only one of the pair moves both
                if self.unique_id < self.partner.unique_id:
                    self.colony()
            else:
                self.disperse()
                if self.remove:
                    return

        if self.territory:
            self.build_dam()

        # Repair abandoned dams in territory
        for dam in self.model.type[Dam]:
            if dam.pos in self.territory and dam.abandoned and dam.repairable:
                dam.abandoned = False
                dam.decay_timer = None
                dam.repairable = False

        # Background mortality
        if np.random.rand() < 0.002:
            self.remove = True
            return

        # Reproduction timer reset
        if self.model.month == 1:
            self.kits_this_year = False
            if self.sex == "F":
                self.breeding_month = self.random.choice([4, 5, 6]) # Produce kits in April, May or June

        # Reproduction logic
        if (self.sex == "F" and
            self.partner and self.partner.partner == self 
            and self.model.month == self.breeding_month and  
            not self.kits_this_year):
            self.reproduce()
            self.kits_this_year = True

        # Age and mortality
        self.age += 1
        if self.age >= self.death_age:
            self.remove = True
            return

    def mate(self, x=None, y=None, max_dist=4000):
        # Find and establish a mate relationship
        potential_mates = []
        for a in self.model.type[Beaver]:
            if(a is not self
               and isinstance(a,Beaver)
               and a.sex!=self.sex
               and (a.partner is None or getattr(a.partner, "remove", False) or a.partner.partner != a)):
                if x is not None and y is not None and max_dist is not None:
                    if a.territory:
                        tx, ty =np.mean([p[0] for p in a.territory]), np.mean([p[1] for p in a.territory])
                    else:
                        tx, ty = a.pos
                    dist = np.sqrt((tx - x) ** 2 + (ty - y) ** 2)
                    if dist > max_dist:
                        continue
                potential_mates.append(a)
        
        if potential_mates:
            mate = self.random.choice(potential_mates)
            self.partner = mate
            mate.partner = self
            self.dispersal_attempts = 0
            return mate
        return None
        
    def disperse(self):
        # Move beaver to a new location using exponential dispersal distance
        mean_dispersal_distance = 1400  # 7km / 5m grid
        cell_width = getattr(self.model.grid, "cell_width", 5)
        water_threshold = 100

        distance = int(np.random.exponential(mean_dispersal_distance))
        angle = np.random.uniform(0, 2 * np.pi)
        dx = int(distance *np.cos(angle))
        dy = int(distance *np.sin(angle))
        x0,y0 = self.pos
        x_new, y_new = np.clip(x0 + dx, 0, self.model.dem.shape[1]-1),  np.clip(y0 + dy, 0, self.model.dem.shape[0]-1)

        distance_to_water = self.model.distance_to_water
        possible_cells = np.argwhere(distance_to_water <= water_threshold)
        
        if possible_cells.size > 0:
            distances = np.linalg.norm(possible_cells - np.array([y_new, x_new]), axis = 1)
            best_idx = np.argmin(distances)
            y_final, x_final = possible_cells[best_idx]
        else:
            x_final, y_final = x_new, y_new

        if self.pos is not None:
            self.model.grid.move_agent(self, (int(x_final), int(y_final)))
        else:
            self.model.grid.place_agent(self, (int(x_final), int(y_final)))

        old_pos = self.pos
        self.form_territory()
        if self.territory:
            self.dispersal_attempts = 0
            return
        else:
            if old_pos is not None and self.pos is not None:
                self.model.grid.move_agent(self, old_pos)
            elif old_pos is not None and self.pos is None:
                self.model.grid.place_agent(self, old_pos)
            return

    def colony(self):
        # Move partner and kits together as a unit
        possible_move = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        valid_move = []
        for pos in possible_move:
            x, y = pos
            if (0<= x < self.model.dem.shape[1] and
                0<= y < self.model.dem.shape[0] and
                self.model.dem[y,x] != 0):
                valid_move.append(pos)
        if not valid_move:
            return
        new_area = self.random.choice(valid_move)
        if self.pos != new_area:
            self.model.grid.move_agent(self, new_area)
            # move partner
        if self.partner and not getattr(self.partner, "remove", False): 
            if self.partner.pos is not None:
                if self.partner.pos != new_area:
                    self.model.grid.move_agent(self.partner, new_area)
            else:
                self.model.grid.place_agent(self.partner, new_area)
            # move kit
        for agent in self.model.grid.get_cell_list_contents([self.pos]):
            if isinstance(agent, (Kit, Juvenile)) and not getattr(agent, "remove", False): 
                if agent.pos is not None:
                    if agent.pos != new_area:
                        self.model.grid.move_agent(agent, new_area)
                else:
                    self.model.grid.place_agent(agent, new_area)



    def form_territory(self):
        # Form territory along suitable waterways
        cell_length = getattr(self.model.grid, "cell_width", 5)
        mean = np.log(3000)  # Mean bankful length of territory is ~3km
        sigma = 0.5
        bank_length = np.random.lognormal(mean=mean, sigma=sigma)
        bank_length = np.clip(bank_length, 500, 20000)  # 0.5 - 20 km range
        territory_cells = max(int(bank_length / cell_length), 1)

        occupied = set()
        for agent in self.model.type[Beaver]:
            if agent is not self and agent.territory:
                occupied.update(agent.territory)

        hsm = self.model.hsm
        distance_to_water = self.model.distance_to_water

        waterway_buffer = 100
        mask =(distance_to_water <= waterway_buffer) & (hsm >=2) & (hsm <=4)
        if occupied:
            occupied_array = np.array(list(occupied))
            mask[occupied_array[:, 1], occupied_array[:, 0]] = 0

        # Create radius mask based on territory size
        radius = max(int(bank_length / cell_length * 1.5), 600)  # Minimum 3km
        cx, cy = self.pos
        y_indices, x_indices = np.indices(mask.shape)
        radius_mask = ((x_indices - cx) ** 2 + (y_indices - cy) ** 2) <= radius ** 2
        mask = mask & radius_mask
        mask[cy, cx] = True
        
        labeled_array, num_features = label(mask)
        territory_label = labeled_array[cy, cx]

        if territory_label == 0:
            # No connected region at beaver's location
            if self.pos is not None:
                self.territory = set()
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
        
            # Only add if not unsuitable or water
            if (labeled_array[y, x] == territory_label and hsm[y, x] in [2, 3, 4]):
                patch.append((x, y))
        
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < labeled_array.shape[0] and
                        0 <= nx < labeled_array.shape[1] and
                        (ny, nx) not in visited and
                        labeled_array[ny, nx] == territory_label and
                        hsm[ny, nx] in [2, 3, 4]):
                        frontier.append((cell_score(ny, nx), ny, nx))

        self.territory = set(patch)
        # Most territories last 4 years, some much longer
        self.territory_abandonment_timer = int(np.random.exponential(48))

        # Assign the same territory to partner
        if self.partner and self.partner.partner == self:
            self.partner.territory = self.territory
            self.partner.territory_abandonment_timer = self.territory_abandonment_timer

        # Assign the same territory to all offspring
        for agent in self.model.grid.get_cell_list_contents([self.pos]):
            if isinstance(agent, (Kit, Juvenile)) and not getattr(agent, "remove", False):
                agent.territory = self.territory
                agent.territory_abandonment_timer = self.territory_abandonment_timer

    def abandon_territory(self):
        # Mark all dams in territory as abandoned
        for dam in self.model.type[Dam]:
            if dam.pos in self.territory and not dam.abandoned:
                dam.abandoned = True
                dam.decay_timer = int(np.random.normal(24, 6))  # ~2 years
                dam.repairable = True

        self.territory = set()
        self.territory_abandonment_timer = None
        

  
        # Move partner and kits with agent
        if self.partner:
            self.partner.territory = set()
            self.partner.territory_abandonment_timer = None
            if self.partner.pos is not None:
                self.model.grid.move_agent(self.partner, self.pos)
            else:
                self.model.grid.place_agent(self.partner, self.pos)

        for agent in self.model.grid.get_cell_list_contents([self.pos]):
            if isinstance(agent, Kit):
                agent.territory = set()
                agent.territory_abandonment_timer = None
                if agent.pos is not None:
                    self.model.grid.move_agent(agent, self.pos)
                else:
                    self.model.grid.place_agent(agent, self.pos)

    def reproduce(self):
        # Create 1-4 kits if conditions are met
        if self.partner is not None and self.pos is not None:
            for _ in range(self.random.randint(1, 4)):
                kit = Kit(self.model, sex=self.random.choice(['M', 'F']))
                kit.territory = self.territory  # Inherit parent's territory
                kit.territory_abandonment_timer = self.territory_abandonment_timer
                self.model.grid.place_agent(kit, self.pos)
                self.model.type[Beaver].append(kit)

    def age_up(self):
        # Kit -> juvenile at age 2 (24 steps), juvenile -> adult at age 3 (36 steps)
        if isinstance(self, Kit) and self.age >= 24: 
            return Juvenile(self.model, sex=self.sex, age=self.age)
        elif isinstance(self, Juvenile) and self.age >= 36:
            return Adult(self.model, sex=self.sex, age=self.age)
        else:
            return self

    def build_dam(self):
        # Build dams on territory waterways
        if not self.territory:
            return

        # Only one beaver per territory should build dams
        if self.territory:
            for agent in self.model.type[Beaver]:
                if (agent is not self
                    and agent.territory == self.territory
                    and agent.unique_id < self.unique_id):
                    return

        transform = self.model.dem_transform
        inv_transform = ~transform
        real_coords = [rasterio.transform.xy(transform, y, x, offset='center') for (x, y) in self.territory]
        territory_shape = MultiPoint(real_coords).convex_hull

        # Find waterways intersecting territory with suitable gradient
        water_in_territory = self.model.waterways[
            (self.model.waterways.intersects(territory_shape)) & 
            (self.model.waterways["gradient"] <= 4)]
        
        if water_in_territory.empty:
            return
        
        dam_attempts = 0
        for idx, segment in water_in_territory.iterrows(): 
            if dam_attempts >= 5:
                break
            gradient = segment["gradient"]
            if gradient == 'NULL' or (isinstance(gradient, float) and np.isnan(gradient)):
                gradient = 0

            channel = segment.geometry  # All points along the channel
            num_points = int(channel.length // 5)  # Every 5m can build
            for fraction in np.linspace(0, 1, num_points):
                if dam_attempts >= 5:
                    break
                point = channel.interpolate(fraction * channel.length)
                col, row = inv_transform * (point.x, point.y)
                x, y = int(round(col)), int(round(row))
                if (x, y) not in self.territory:
                    continue

                existing_dams = self.model.grid.get_cell_list_contents([(x, y)])
                if any(isinstance(a, Dam) for a in existing_dams):
                    continue
            
                temp_dam = Dam(self.model, (x, y), depth=None)
                flood_layer = temp_dam.flood_fill()
                if temp_dam.flood_land():
                    if temp_dam.pos != (x, y):
                        self.model.grid.place_agent(temp_dam, (x, y))
                    self.model.type[Dam].append(temp_dam)
                    self.dam = temp_dam
                    flooded_indices = np.argwhere(temp_dam.flooded_area == 1)
                    for r, c in flooded_indices:
                        if (0 <= r < self.model.hsm.shape[0] and 0 <= c < self.model.hsm.shape[1]):
                            self.model.hsm[r, c] = 6 
                    return
                else:
                    dam_attempts += 1

class Kit(Beaver):
    # Kits move with group, can't pair or reproduce, age up at 24 months
    
    def __init__(self, model, sex=None, age=0):
        super().__init__(model, sex=sex, age=age)

    def step(self):
        # Background mortality
        if np.random.rand() < 0.002:
            self.remove = True
            return

        self.age += 1
        if hasattr(self, "death_age") and self.age >= self.death_age:
            self.remove = True
            return

        # Kits die if no adults in cell
        adults_in_cell = [a for a in self.model.grid.get_cell_list_contents([self.pos]) 
                         if isinstance(a, Adult) and not getattr(a, "remove", False)]
        if not adults_in_cell:
            self.remove = True
            return

        # Age up if applicable
        new_self = self.age_up()
        if new_self is not self:
            self.remove = True
            self.model.grid.place_agent(new_self, self.pos)
            self.model.type[Beaver].append(new_self)
            return


class Juvenile(Beaver):
    # Juveniles can stay in natal colony for longer then disperse, pair and reproduce, age up at 36 months
    
    def __init__(self, model, sex=None, age=0):
        super().__init__(model, sex=sex, age=age)
        self.helper_timer = int(np.clip(np.random.exponential(6), 0, 24))

    def step(self):
        super().step()
        if self.remove:
            return
        
        # Stay as helper for a period
        if self.helper_timer > 0:
            self.helper_timer -= 1
            return

        # Age up if applicable
        new_self = self.age_up()
        if new_self is not self:
            self.remove = True
            self.model.grid.place_agent(new_self, self.pos)
            self.model.type[Beaver].append(new_self)
            return


class Adult(Beaver):
    # Adults have full beaver behaviour - pairing, movement, reproduction, dam building
    
    def __init__(self, model, sex=None, age=0):
        super().__init__(model, sex=sex, age=age)
    
    def step(self):
        super().step()  # Call base beaver logic
        if self.remove:
            return
        
class Dam(Agent):
    # Dam agent that creates flooding and can be abandoned or decay
    
    def __init__(self, model, pos, depth):
        super().__init__(model)
        self.pos = pos
        self.dam_remove = False
        self.abandoned = False
        self.decay_timer = None
        self.repairable = False

        if depth is None:
            mu, sigma = 1.6, 0.44  # Flooding depth
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
        # Calculate flood extent using DEM and dam depth
        if self.pos is None:
            return None
        x0, y0 = self.pos
        dem = self.model.dem
        flood_layer = zeros(dem.shape, dtype=int)
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
                for r_adj, c_adj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:                        
                    neighbour = (r + r_adj, c + c_adj)
                    if (0 <= neighbour[0] < dem.shape[0] and 
                        0 <= neighbour[1] < dem.shape[1] and 
                        neighbour not in assessed):
                        to_assess.add(neighbour)
        self.flooded_area = flood_layer
        return flood_layer

    def flood_land(self):
        # Check if flooding affects non-water habitat
        if self.flooded_area is None:
            return False
        hsm = self.model.hsm
        flooded_indices = np.argwhere(self.flooded_area == 1)
        for r, c in flooded_indices:
            if 0 <= r < hsm.shape[0] and 0 <= c < hsm.shape[1]:
                if hsm[r, c] != 6:
                    return True
        return False
    
    def step(self):
        # Handle dam decay when abandoned
        if self.abandoned:
            if self.decay_timer is not None:
                self.decay_timer -= 1
                if self.decay_timer <= 0:
                
                    flooded_indices = np.argwhere(self.flooded_area == 1)
                    for r, c in flooded_indices:
                        if 0 <= r < self.model.hsm.shape[0] and 0 <= c < self.model.hsm.shape[1]:
                            self.model.hsm[r, c] = 0  # Reset flooded cells
                    self.flooded_area = None
                    self.dam_remove = True
                    self.repairable = False
    