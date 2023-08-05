import networkx as nx
import numpy as np
import time

from .strategies import cooperate, defect, tit_for_tat, generous_tit_for_tat, win_stay_lose_shift
from .initialize_countries import *

from itertools import combinations
from .payoff_functions import default_payoff_functions, traditional_payoff_functions, selfreward

from .enums import to_outcome, Action, random_action

class Tournament:

    def __init__(
        self, 
        countries, 
        max_rounds, 
        strategy_list, 
        payoff_functions=default_payoff_functions, 
        distance_function = lambda d: d, 
        surveillance_penalty = True,
        penalty_dict = {cooperate: 1, defect: 1, tit_for_tat: 0.95, generous_tit_for_tat: 0.95, win_stay_lose_shift: 0.95}, 
        noise = 0
    ):
        """
        parameters:
            - countries: Country
            - max_rounds: int
            - strategy_list: list
            - payoff_functions: dict
        """
    
        self.max_rounds: int = max_rounds
        self.strategy_list: list = strategy_list
        self.payoff_functions: dict = payoff_functions
        self.surveillance_penalty = surveillance_penalty
        self.penalty_dict = penalty_dict
        self.noise = noise
        
       
        self.graph = self.init_graph(countries, self.payoff_functions)
        

        self.fitness_results = np.zeros((len(self.countries()), max_rounds))
        self.evolution = [] # Todo: add evolution
        

        self.round = 0
        self.is_done = False
        
        self.summed_fitness_history = None
       
    @staticmethod
    def init_graph(countries, payoff_functions):
        """
        initialize the graph (form the NetworkX library), that is used to store
        data from the simulation. Nodes in this graph store countries, and edges
        store the data associated with games between these countries (history, 
        payoff values, distance).
        
        parameters:
            - countries: list of Country, countries that take part in the Tournament
        """
      
        graph = nx.DiGraph() 
        
        
        for country in countries:
            graph.add_node(country)
      
        for c1, c2 in combinations(countries, 2):
           
            RR = (payoff_functions['R'](c1,c2), 
                  payoff_functions['R'](c2,c1))
            PP = (payoff_functions['P'](c1,c2), 
                  payoff_functions['P'](c2,c1))
            TS = (payoff_functions['T'](c1,c2), 
                  payoff_functions['S'](c2,c1))
            ST = (payoff_functions['S'](c1,c2), 
                  payoff_functions['T'](c2,c1))
            
            
            
            

            graph.add_edge(
                c1, 
                c2,
            
                history_1 = [], # list to accumulate tuples of actions
                history_2 = [],
                RR = RR, 
                PP = PP,
                TS = TS,
                ST = ST
            )
        
        return graph
    
    
    
    def countries(self):
        """
        return:
            - countries partaking in this tournament
        """
        return self.graph.nodes()
    
    

    def init_strategies(self, countries = None, strategy = None):
        """
        initizalize strategy for given countries. 
        
        parameters:
            countries: list or None, countries to change strategy, if None, then all countries are used
            strategy: list or None, stratey to be given to countries, if None, then echt country
                      is randomly assigned a strategy from self.strategy_list.
        
        side effects:
            - changes the countries strategy
            
        example:
            >>> tournament = Tournament(...)
            >>> tournament.init_strategies(china, cooperate)
        """
        assert self.round == 0
        
        if isinstance(countries, Country):
            countries = [countries]
        
        countries = countries or self.countries()
        
        for country in countries:
            country.change_strategy(
                    0, strategy or np.random.choice(self.strategy_list)
                    )
        
        
    def init_fitness(self, init_fitnes_as_m, countries = None):
        """
        initizalize the fiteness for each country
        
        parameters:
            - init_fitnes_as_m: bool, if countries should start out with fitness
                                equal to their market size.
        
        side effects:
            - changes `fitness` and `fitness_history` for all countries
        """
        countries = countries or self.countries()
        
        for country in countries:
            # todo: think if this logic has a better place in the country class itself..
            country.fitness = country.m if init_fitnes_as_m else 0
            country.fitness_history = [country.fitness]
        
    def countries_per_strategy_dict(self):
        """
        return:
            - dictionary where the keys are strategies and the values the number of countries that play this strategy
            
        example:
            >>> tournament = Tournament(XXX)
            >>> tournament.init_strategies(None, cooperate)
            >>> tournament.countries_per_strategy_dict()
            {cooperate: 100, defect: 0}
        """
        d = {strategy: 0 for strategy in self.strategy_list}
        for country in self.countries():
            d[country._strategy] += 1
        return d
        
        
        
    def one_strategy_left(self, strategy_n_countries_dict):
        """
        returns:
            - True if there is only one strategy left in the simulation
            
        example:
            >>> tournament = Tournament(XXX)
            >>> tournament.init_strategies(None, cooperate)  # set al countries to cooperating strategy
            >>> tournament.one_strategy_left()
            True
        """
        for value in strategy_n_countries_dict.values():
            if value == len(self.countries()):
                return True
        return False

    
    def change_a_strategy(self, mutation_rate, round_num):
        """
        Change the strategy of a random country, to become the strategy of a
        'winning country', that was selected with probabilites proporitonal to 
        the fitness. This way strategies that do well in the tournament will
        spread through countries.
        
        Sometimes, in stead of the above, there will be a change in strategy
        that is entirely random.
        
        parameters:
            - mutation_rate: probabilitie of a random strategy change
            
        side effect:
            - changes a countries strategy
            
        """
        
        country_list = list(self.countries())

        N = len(country_list)
        
       
        elimiation_idx = np.random.randint(N)
        losing_country = country_list[elimiation_idx]
        losing_strategy = losing_country.get_current_strategy()
        
  
        mutation = bool(np.random.binomial(1, mutation_rate))
        if mutation:
            # in stead of changing a strategy according to the rules of the simmulation
            # we sometimes have a random mutation
            winning_strategy = np.random.choice(self.strategy_list)
            winning_country = 'random_mutation'
        else:
            
            fitness_scores = [country.fitness for country in country_list]
            fitness_scores_non_neg = [max(0, fitness) for fitness in fitness_scores]
            total_fitness = sum(fitness_scores_non_neg)
            probabilities = [fitness_scores_non_neg[j]/total_fitness for j in range(N)] # errors if total fitness becomes negative...
            

            reproduce_idx = np.random.choice(range(N), p=probabilities)
            winning_country = country_list[reproduce_idx]
            winning_strategy = winning_country.get_current_strategy()
        
     
        losing_country.change_strategy(round_num, winning_strategy)
        print(f'  strategy change: {losing_country} {losing_strategy.name} -> {winning_strategy.name} ({winning_country})')

        return losing_country, winning_strategy 
    
    def play_prisoners_dilema(self, country_1, country_2, game):
        """
        parameters:
            - coutry_1, country_2: Country
            - game: dict, data associated with the game between countr_1 and country_2 
            
        side effects:
            - appends history of game
            - changes fitness of country_1 and country_2
            
     
        """
                   
            
     
        action_1 = country_1.select_action(game['history_1'], game['history_2'], self.noise)
        action_2 = country_2.select_action(game['history_2'], game['history_1'], self.noise)

        game['history_1'].append(action_1)
        game['history_2'].append(action_2)

        outcome = to_outcome(action_1, action_2)
        Δfitness_1,  Δfitness_2 = game[outcome] #self.graph.get_edge_data(country_1, country_2)[outcome]
        
        if self.surveillance_penalty:
            # to simmulate the effort that it takes to take a certain strategy
            # for some strategies a penalty is added, so that only part of the
            # change in fitness is really received.
            Δfitness_1 *= self.penalty_dict[country_1.get_current_strategy()]
            Δfitness_2 *= self.penalty_dict[country_2.get_current_strategy()]
            

        country_1.fitness += Δfitness_1
        country_2.fitness += Δfitness_2

        
        return Δfitness_1,  Δfitness_2, outcome
        
    def check_all_strategies_initialized(self):
        """
        example:
            >>> tournament = Tournament(XXX)
            >>> tournament.check_all_strategies_initialized()
            False
            >>> tournament.init_strategies()
            >>> tournament.check_all_strategies_initialized()
            True
        """
        for country in self.countries():
            if country.get_current_strategy() is None:
                print(f'WARNING: {country} has no initizalized strategy')
        
    def play(self, self_reward, playing_each_other, nr_strategy_changes, mutation_rate):
        """
        parameters:
            - self_reward: function or None, None indicates countries do not get 
                           reward from their internal market, if they do, self_reward
                           should be the function of the country that gives them 
                           self_reward.
            - playing_each_other: bool, if countries play prisoners-dilema's with each
              other, and get/lose fitness from this
            - nr_strategy_changes: int, number of strategy-changes that occura
              after each round
              
        example:
            >>> tournament = Tournament(XXX)
            >>> tournament.init_strategies()
            >>> tournament.play(XXX)
            XXXXXXX
        """
        if self.is_done:
            print("WARNING: you are playing a tournament that has already been played. This will accumulate more"\
                  "data in the graph, which is probably incorrect. You probably want to re-initalize the tournament and"\
                  "countries, or refresh the kernel")
        
        strategies_initialized = self.check_strategies_initialized()
        if not strategies_initialized:
            print(f'All countries mus have initialized strategies, this can be done using the init_strategies method')
            return
        
  
        for i in range(self.max_rounds):
            #start_t = time.time()
            
            self.round += 1
            print(f'=== ROUND {self.round} ===')
            
            if self_reward:
                for country in self.countries():
                    country.fitness += self_reward(country)
            
            if playing_each_other:
                for country_1, country_2, data in self.graph.edges(data=True): 
     
                    self.play_prisoners_dilema(country_1, country_2, data)
                    
                    
            for _ in range(nr_strategy_changes):
                losing_country, winning_strategy = self.change_a_strategy(mutation_rate, self.round)
                
            
            for country in self.countries():
                country.fitness_history.append(country.fitness)
                
            if self.one_strategy_left(self.countries_per_strategy_dict()):
                print(f'The process ended in {i+1} rounds\n Winning strategy: {list(self.countries())[0].get_current_strategy().name}')
                break
            
           
            if self.round % 10 == 0:
                for (s,i) in self.countries_per_strategy_dict().items():
                    print(f'    {s.name}: {i}')


        self.is_done = True
        
            
    def check_strategies_initialized(self):
        """
        check if all countries have a strategy.
        """
        for country in self.countries():
            if country.get_current_strategy() not in self.strategy_list:
                print(f'country {country.name} has strategy {country.get_current_strategy()} which is not in the strategy_list')
                return False
        return True

    def fitness_history_sum_list(self, selecting=[], filtering = []):
        """
        return the fitness of all contries summed, in a list of rounds.
        """
    
    
        if selecting:
            countries=selecting
        elif filtering:
            countries = [country for country in self.countries if not country in filtering]
        else:
            countries = list(self.countries())
    
        fitness_histories = [c.fitness_history for c in countries]
        ls = [sum(fitnesses) for fitnesses in zip(*fitness_histories)]
        
        self.summed_fitness_history = ls


    @classmethod
    def create_play_tournament(cls, 
                 countries, 
                 max_rounds, 
                 strategy_list=[defect, cooperate], 
                 payoff_functions=default_payoff_functions, 
                 distance_function = lambda d: d, # defaults to just the identity
                 surveillance_penalty = True,
                 self_reward = selfreward, #default function
                 playing_each_other=True,
                 nr_strategy_changes = 1,
                 mutation_rate =0.1,
                 init_fitnes_as_m=False,
                 noise = 0
                 ):
        """
        Create a tournament, initialize al the variables of the countries and
        then play the tournament.
        
        parameters:
            - countries: list, countries that take part in the tournament
            - max_rounds: int, maximum number of rounds, after which the tournament will stop
            - strategy_list: list, strategies that are played in the tournament
            - payoff_functions: functions to compute the changes in fitness, e.g. `default_payoff_functions` or `traditional_payoff_functions`
            - distance_function: function to rescale the distance. e.g. `lambda d:d` for linear scaleing and `lambda d: math.log(1+d)` for log-scaling
            - surveillance_penalty: bool, if countries should be penalized for playing certain strategies
            - self_reward: function or None, a function if countries should get reward from their internal market each round, 
                           otherwize, if countries should not get reward from their internal market, None
            - playing_each_other: bool, if countries should play prisoners delemma's with each other, set to false to create a control-group
            - nr_strategy_changes: int, number of strategy changes after eacht round
            - mutation_rate: probability that a strategy change is random
            - init_fitnes_as_m: bool, if countries start with self.fitness==self.m or self.fitness==0
        
        returns:
            the tournament object, with data from the simulation inside the
            graph attribute.

        """
        tournament = cls(countries, 
                 max_rounds, 
                 strategy_list, 
                 payoff_functions=payoff_functions, # rewards that countries get, defaults to the functions described in the paper.
                 distance_function = distance_function, # defaults to just the identity, if one wanted that distances get less important the larger they are, one could use the sqrt.
                 surveillance_penalty = surveillance_penalty,
                 noise=noise
                 )
        tournament.init_strategies()
        tournament.init_fitness(init_fitnes_as_m=init_fitnes_as_m)
        
        tournament.play(self_reward, playing_each_other, nr_strategy_changes, mutation_rate)
        
        
        tournament.fitness_history_sum_list()
        
        return tournament
    


        
    
