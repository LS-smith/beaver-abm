from mesa import Agent

class Beaver(Agent):
    """Base Beaver Class"""

    def __init__(self, unique_id, model, sex=None, cell=None, age=0):
        """
		* Initialise and populate the model
		"""
        super().__init__(unique_id, model) 
        self.sex = sex if sex else model.random.choice(['M', 'F'])
        self.partner = None
        self.age = age
        self.reproduction_timer = 0
        self.remove = False # mark for removal

    def step(self):
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


        # move together if paired, else move alone
        if self.partner and self.partner.partner == self:
            if self.unique_id < self.partner.unique_id:  # only one of the pair moves both
                self.move(together=True)
        else:
            self.move(together=False)

    def move(self, together=False):
        possible_move = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        if possible_move:
            new_area = self.random.choice(possible_move)
            self.model.grid.move_agent(self, new_area)
            if together and self.partner:
                if not getattr(self.partner, "remove", False):  # check if partner is not marked for removal
                    self.model.grid.move_agent(self.partner, new_area)
       

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

    def move(self, together=False):
        neighbours = self.model.grid.get_cell_list_contents([self.pos]) # move with colony
        adults = [a for a in neighbours if isinstance(a, Adult)] #find adulgt in same cell
        if adults:
            self.model.grid.move_agent(self, adults[0].pos) # move to lead adults new cell - if no adult dont move!
        
         #TODO: finish later, should only move with parents or die - think this will mess up when parent dead so add in that 


    def step(self): 
        self.move() # specific movement logic - move with colony
        self.age += 1  

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
        self.move()
        self.age += 1  

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
