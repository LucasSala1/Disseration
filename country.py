from .enums import random_action
import numpy as np

class Country:
    
    "Countries in the simulation"
    
    def __init__(self, name, e, i): 
        
        # Variables that stay the same during simulations
        self.name = name
        self.e = e
        self.i = i
        self.self_reward = None
        
        # State variables, not yet initialized since that will be done in the tournament
        self.fitness = None 
        self.fitness_history = []
    
        # private attributes, they should only be changed with `change_strategy`
        self._strategy = None 
        self._evolution = []
        
      
    def __str__(self):
        return f'<{self.name}>'
        
    def __repr__(self):
        return f'<{self.name}>'
    

    
    def change_strategy(self, round_num, strategy):
        """
        parameters:
            - round_num: int, round number when the change occured
            - strategy: new strategy that the country adopts
        
        side effects:
            - set self._strategy to the new strategy
            - appends self.evolution
        """
        self._strategy = strategy
        self._evolution.append((round_num, strategy))
        
    def select_action(self, selfmoves, othermoves, noise_threshold):
        r = np.random.uniform()
        if r < noise_threshold:
            # there is a chance of {noise_threshold} that this action will 
            # be randomly selected
            return random_action()
        else:
            return self._strategy(selfmoves, othermoves)
    
    def get_current_strategy(self):
        """
        returns:
            - current strategy
        """
        return self._strategy
