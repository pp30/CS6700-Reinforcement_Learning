# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:38:10 2020

@author: Pawan Prasad
"""

import gym
import numpy as np
from matplotlib import pyplot as plt
import tqdm


class SarsaPolicy:      

    def sarsa_update(self, curr_state, curr_action, reward, next_state, next_action, Q, alpha, gamma):
        '''
        ------------------------------------------------------------------------------------------
        Update for Q Table using current state, current action, reward obtained, next state, next action
        ------------------------------------------------------------------------------------------
        
        '''
        error = reward + (gamma*Q[next_action][next_state[0],next_state[1]]) - Q[curr_action][curr_state[0],curr_state[1]]
        Q[curr_action][curr_state[0],curr_state[1]] = Q[curr_action][curr_state[0],curr_state[1]] + alpha*error
        
        return Q




    def sarsa_choose_action(self,epsilon, state, Q, env):
        '''
        ---------------------------choose action Epsilon-greedily---------------------------------------------
        '''
      
        if np.random.uniform(0,1) < epsilon:
            action = env.random_action()
 
        else:
            action = np.argmax(Q[:,state[0],state[1]])
        return action
    
    
    def sarsa(self, gamma, alpha, epsilon, episodes, env, goal):
         
        ''' 
        ----------------------------------The sarsa algorithm implementation-------------------------------------------
        '''
        goal_pos = env.goal_pos(goal)
    
        Q = np.random.rand(4, 12, 12)
       
        steps = list(range(episodes))
        rewards = list(range(episodes))
        
        for i in range(episodes):
            steps[i] = 0
            rewards[i] = 0
        
        for i in tqdm.tqdm(range(episodes)):
                        
                    done = False
                    
                    curr_state = env.reset()
                    
                    curr_action = self.sarsa_choose_action(epsilon, curr_state, Q, env)
        
                    
                    count=0
                    while done == False:
                        
                        
                        next_state, reward, done, info = env.step(curr_action)
                        
                        if (done==True): 
                            
                            count =count+1  #this is solely for debugging
                            
                        
                        
                        next_action = self.sarsa_choose_action(epsilon, next_state, Q, env)
        
    
                        Q = self.sarsa_update(curr_state, curr_action, reward, next_state, next_action, Q, alpha, gamma)
                                        
                       
                        curr_state = next_state
                        curr_action = next_action
                       
                        steps[i]+=1
                        rewards[i] += reward
                       
        
        
        return rewards, steps, Q

    def sarsa_avg_plot(self, avg_reward, steps, episodes, goal):
            '''
            -----------------------avg reward/avg steps VS episodes curves -----------------
            '''
         
            fig1=plt.figure(figsize=(10,6)).add_subplot(111)
            fig2=plt.figure(figsize=(10,6)).add_subplot(111)
          
            colors = ['g', 'r', 'k', 'b', 'y','m', 'c']
            fig1.plot(range(episodes), avg_reward, colors[0], label = " Average reward " )
            fig2.plot(range(episodes), steps, colors[1], label = " Steps")
            
            fig1.title.set_text('Sarsa' + ' : Average reward VS episodes for 50 runs: Goal =  '+ goal)
            fig1.set_ylabel('Average Reward')
            fig1.set_xlabel('episodes')
            fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            if goal == 'C':
                fig1.axis(ymin = -100 ,ymax = 10)
            
            fig2.title.set_text('Sarsa' + ' : Average steps VS episodes for 50 runs: Goal = '+ goal)
            fig2.set_ylabel('Steps')
            fig2.set_xlabel('episodes')
            fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
          
            plt.show()
 
    def sarsa_plot_policy(self, goal_pos, policy):
        '''
       ------------------------------ Policy map for 12 x 12 grid-------------------------------------------------- 
        '''
        print("\nPolicy Map\n")
        plt.rcParams['figure.figsize'] = [7,7]
        fig, ax = plt.subplots()
        
        ax.matshow(policy)
    
        for i in range(12):
            for j in range(12):
                if (j,i) == env.goal_coord :
                    
                    ax.text(i,j,'G', va='center', ha='center')
                else:
                    
                    c = int(policy[j,i])
                    arrows = {0:'↑', 1:'➜', 2:'←', 3:'↓' }
                    ax.text(i, j, arrows[c], va='center', ha='center')

        plt.show()   

        return             


if __name__ == '__main__':
    
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.1
    episodes = 300
    runs = 10
    
    env = gym.make('gym_pdlwrld:pdlwrld-v0')
    
  
    sr = SarsaPolicy()
    
    '''
   --------------------------------------NOTE: While running code set goal variable manually to 'A' 'B' or 'C'------------------------------------------------------------------
    
    '''
    goal = 'A'

    REW = list(range(runs))
    STEP = list (range(runs))
    Qruns = list (range(runs))
    
    for i in range(runs):
            REW[i] = 0
            STEP[i] = 0
            Qruns[i] = 0    
    
    R = list(range(episodes))
    S = list(range(episodes))
    Q = np.random.rand(4, 12, 12)
    
    for i in tqdm.tqdm(range(runs)):
        R,S,Q = sr.sarsa(gamma, alpha, epsilon, episodes, env, goal)
        REW[i]= R
        STEP[i] = S
        Qruns[i] = Q
   
    
    r_avg = list(range(episodes))
    s_avg =list(range(episodes))
    
    for i in range(episodes):
            r_avg[i] = 0
            s_avg[i] = 0
              
    
    for i in range(episodes):    
        for j in range(runs):
            r_avg[i] += REW[j][i]   
            s_avg[i] += STEP[j][i]
    
    #computing average across 50 runs
    for i in range(episodes):
      
      r_avg[i] = r_avg[i]/runs
      s_avg[i] = s_avg[i]/runs
       
    
    
    
    sr.sarsa_avg_plot(r_avg, s_avg, episodes, goal)
    
    policy = np.zeros([12,12])
    
    val = {0:0, 1:0, 2:0, 3:0}
    
    #print("\nentering policy domain")
  #finding optimum action (policy) 
    for i in range(12):
        for j in range(12):
            for k in range(len(Qruns)):
                val[np.argmax(Qruns[k], axis=0)[i,j]]+=1
              
            max_action = max(val, key=val.get)
            val = {0:0, 1:0, 2:0, 3:0}
            policy[i,j] = max_action
            
    sr.sarsa_plot_policy(goal, policy)        
      
