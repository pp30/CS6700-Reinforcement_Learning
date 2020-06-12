# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:35:48 2020

@author: Pawan Prasad
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random


import numpy as np

class PdlwrldEnv(gym.Env):
  metadata = {'render.modes': ['human']}
    
  def __init__(self):
      
   self.puddle = [(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),
                  (3,3),(4,3),(5,3),(6,3),(7,3),(8,3),
                  (8,4),(8,5),(8,6),(8,7),
                  (7,7),(6,7),
                  (6,8),(5,8),(4,8),(3,8)] #(6,7),
   
   self.puddler = [(3,4),(3,5),(3,6),(3,7),
                   (4,4),(5,4),(6,4),(7,4),
                   (7,5),(7,6),
                   (6,6),(5,6),
                   (5,7),(4,7)]
                   
                   
   self.puddlest = [(4,5),(4,6),
                    (5,5),(6,5)]    
   
   self.height = 12
   self.width = 12
   self.action_space = spaces.Discrete(4)
   self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))
   
   self.moves = {
                0: (-1, 0),  # North
                1: (0, 1),   # East
                2: (1, 0),   # South
                3: (0, -1),  # West
                }

   self.wind_move = (0,1)
   #self.goal_coords = [[0,11],[2,9],[6,7]]
   self.Done = None
   self.R= None
   self.S = None
   self.goal =None
   self.goal_coord = None
   
   self.reset()
   
  def random_action(self):
        self.action = np.random.choice([0,1,2,3])
        return self.action 
  
  def goal_pos(self,goal):
      if goal == 'A':
          self.goal_coord = (0,11)
      elif goal == 'B':
          self.goal_coord = (2,9)
      elif goal == 'C':
          self.goal_coord = (6,7)
    
  def other_possible_actions(self, action):
      self.ar =[]
      for i in range(4):
          if i != action:
              self.ar.append(i)
      return self.ar
        
  def reward(self):
      r = 0
      done =False
      
      if self.S == self.goal_coord:
          r = 10
          
          done = True
          #print("from done classsss")
      elif self.S in self.puddle:
          r = -1
      elif self.S in self.puddler:
          r = -2
      elif self.S in self.puddlest:
          r = -3
      #if self.goal_coord != (6,7): 
          #r=-1
      
      return r, done 
   
  def step(self, action):
      
    un = random.uniform(0, 1)
    un_wind = random.uniform(0,1)
    
    if un > 0.9:
        i = np.random.randint(low=0,high =3)    
        new_set = self.other_possible_actions(action)
        
        (x,y) = self.moves[new_set[i]]
        
        if un_wind > 0.5:
            (x,y) = x + self.wind_move[0], y+self.wind_move[1]
      
    else:
        (x,y) = self.moves[action]
        if un_wind > 0.5:
            (x,y) = x + self.wind_move[0], y+self.wind_move[1]
            
    if self.goal_coord == (6,7):
        #if self.S == (6,7): #goal position C
            self.S = self.S[0] - self.wind_move[0], self.S[1] - self.wind_move[1]
        
    self.S = self.S[0] + x, self.S[1] + y
    
    

    self.S = max(0, self.S[0]), max(0, self.S[1])
    self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))
    
    self.R, self.Done = self.reward()

    return self.S, self.R, self.Done, {}              
        
        
      
      
      
      
    
  
    
  def reset(self):
      start_set = [(5,0),(6,0),(10,0),(11,0)]
      i = np.random.randint(low=0,high=4)
      self.S = start_set[i]
      return self.S
  
  
      
    
    
  def render(self, mode='human'):
      pass
    
    
  def close(self):
      pass
    
    
    
