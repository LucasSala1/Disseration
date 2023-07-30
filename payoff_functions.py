import math as m


def selfreward(country):
    '''calculates how much reward a country gets from its own internal market'''
    return (country.e) / (country.i)

def reward(country, other):
    '''calculates how much reward would change fitness'''
    return (country.e+other.e*0.025) / (country.i + country.i*0.1)  

def temptation(country, other):
    '''calculates how much temptation would change fitness'''
    return (country.e) / country.i

def sucker(country, other):
    '''calculates how much sucker would change fitness'''
    return (country.e*0.025) / (country.i + country.i*0.1)

def punishment(country, other):
    '''calculates how much punishment would chage fitness'''
    return 0

default_payoff_functions = {
        'R': reward,
        'T': temptation,
        'S': sucker,
        'P': punishment,
        'self_reward': selfreward
        }

traditional_payoff_functions = {
        'R': lambda *args: 3,
        'T': lambda *args: 5,
        'S': lambda *args: 0,
        'P': lambda *args: 1
        }
