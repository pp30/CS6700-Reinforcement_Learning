# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:38:10 2020

@author: Pawan Prasad
"""

import gym
import numpy as np
from matplotlib import pyplot as plt


class SarsaPolicy_LAMBDA:     

    def sarsa_update(self, curr_state, curr_action, reward, next_state, next_action,Q, alpha, gamma, E):
        '''
        ------------------------------------------------------------------------------------------
        Update for Q Table using current state, current action, reward obtained, next state, next action and Eligibilty trace
        ------------------------------------------------------------------------------------------
        
        '''
        error = reward + (gamma*Q[next_action][next_state[0],next_state[1]]) - Q[curr_action][curr_state[0],curr_state[1]]
        Q[curr_action][curr_state[0],curr_state[1]] = Q[curr_action][curr_state[0],curr_state[1]] + alpha*error* E[curr_action][curr_state[0],curr_state[1]]
        #here we are also taking into account E for update 
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
    
    
    def sarsa(self, gamma, alpha, epsilon, episodes,env,goal, lmbda ):
         
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
        
        #MAX_STEPS = 1000
        
        for i in range(episodes):
                        
                    done = False
                    
                    curr_state = env.reset()
                    
                    curr_action = self.sarsa_choose_action(epsilon, curr_state, Q, env)
        
                    E = np.random.rand(4, 12, 12)
                    
                    
                    count=0
                    
                        
                    while done == False:
                            
                            # Get the next state and reward based on the current state and action perfomed
                            next_state, reward, done, info = env.step(curr_action)
                            #print("Reward for step:",count," episode",i,": ",reward)
                            if (done==True): 
                                #print()
                                count =count+1   #this is solely for debugging
                                #print("from done trueeeeee",next_state,count,"episode number",i)
                            
                            # Perform next action based on the above next state, Q function using epsilon greedy method
                            next_action = self.sarsa_choose_action(epsilon, next_state, Q, env)
            
                            
                             #Taking into account eligibility traces since sarsa lambda
                            
                            E = E * (gamma*lmbda)
                            E[curr_action][curr_state[0],curr_state[1]] += 1
                            
                            #here we are also taking into account E for update                                         
                            Q = self.sarsa_update(curr_state, curr_action, reward, next_state, next_action,Q, alpha, gamma, E)
                                            
                            
                            curr_state = next_state
                            curr_action = next_action
                            
                            
                            steps[i]+=1
                            
                            rewards[i] += reward
                       
        
                        #if curr_state == goal_pos:
                            # print("Steps =======================", steps[episode])
                            # print("reward=======================", avg_reward[episode])
                           # break
                    #print("rewards",[i],":",rewards[i])       
        
        return rewards, steps, Q

    def sarsa_avg_plot(self, avg_reward, steps, episodes, goal,lmbda):
            
            '''
            -----------------------avg reward/avg steps VS episodes curves for various lambda-----------------
            '''
            fig1=plt.figure(figsize=(10,6)).add_subplot(111)
            fig2=plt.figure(figsize=(10,6)).add_subplot(111)
            
            colors = ['g', 'r', 'k', 'b', 'y','m', 'c']
            fig1.plot(range(episodes), avg_reward, colors[0], label = " Average Reward " )
            fig2.plot(range(episodes), steps, colors[1], label = " Steps")
            
            fig1.title.set_text('Sarsa'+ '($\lambda$)'+ ' : Average reward VS episodes for 50 runs: lambda = '+ str(lmbda) + ' goal = '+goal)
            fig1.set_ylabel('Average Reward')
            fig1.set_xlabel('Episodes')
            fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
               
            fig2.title.set_text('Sarsa' + '($\lambda$)'+' : Average steps VS episodes for 50 runs: lambda = '+ str(lmbda) + ' goal = '+ goal)
            fig2.set_ylabel('Steps')
            fig2.set_xlabel('Episodes')
            fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
            plt.show()
            
          
    def sarsa_plot_policy(self, goal_pos, policy, lmbda):
        '''
       ------------------------------ Policy map for 12 x 12 grid-------------------------------------------------- 
        '''
        
        print("\n--------------POLICY MAP for LAMBDA: ", lmbda,"\n")
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
    
    
    
    
    def sarsa_avg_plot_all(self, rewards_all, steps_all, episodes, goal,lambda_list):
     
        '''
        -----------------------ALL IN ONE PLOT: avg reward/avg steps VS episodes curves for various lambda-----------------
        '''
        
        fig1=plt.figure(figsize=(10,6)).add_subplot(111)
        fig2=plt.figure(figsize=(10,6)).add_subplot(111)

        colors = ['b', 'r', 'g', 'm', 'y','k', 'c']

        for i in range(len(rewards_all)):
            fig1.plot(range(25,episodes), rewards_all[i][25:], colors[i], label = "lambda = " + str(lambda_list[i]) )

        for i in range(len(steps_all)):
            fig2.plot(range(25,episodes), steps_all[i][25:], colors[i], label = "lambda = " + str(lambda_list[i]) )

        
        fig1.title.set_text('Sarsa_lambda plots of Average reward VS episodes comparison (epsiode 25-700)')
        fig1.set_ylabel('Average Reward')
        fig1.set_xlabel('episodes')
        fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        
        fig2.title.set_text('Sarsa_lambda plots of Average steps VS episodes comparison (episode 25-700)')
        fig2.set_ylabel('Steps')
        fig2.set_xlabel('episodes')
        fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

      
        plt.show()
            

    



if __name__ == '__main__':
   
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.1
    episodes = 700
    runs = 50
    lambda_list = [0, 0.3, 0.5, 0.9, 0.99, 1.0]
    env = gym.make('gym_pdlwrld:pdlwrld-v0')
    
    
    sr = SarsaPolicy_LAMBDA()
    
    '''
   -------------------------------- NOTE: While running code set goal variable manually to 'A' 'B' or 'C'------------------------------------------------------------------
    
    '''
    goal = 'C'
    
    r_avg_comp = list(range(len(lambda_list)))
    s_avg_comp = list(range(len(lambda_list)))
    
   
   
    
    for p in range(len(lambda_list)) :
        lmbda = lambda_list[p]
        
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
            
        for i in range(runs):
            R,S,Q = sr.sarsa(gamma, alpha, epsilon, episodes, env, goal,lmbda)
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
       
        r_avg_comp[p] = r_avg
        s_avg_comp[p] = s_avg
    
    
        sr.sarsa_avg_plot(r_avg, s_avg, episodes, goal,lmbda)
    
        policy = np.zeros([12,12])
    
        val = {0:0, 1:0, 2:0, 3:0}
    
    
        #finding optimum action (policy)  
        for i in range(12):
            for j in range(12):
                for k in range(len(Qruns)):
                    val[np.argmax(Qruns[k], axis=0)[i,j]]+=1
                    
                max_action = max(val, key=val.get)
                val = {0:0, 1:0, 2:0, 3:0}
                policy[i,j] = max_action
                
        sr.sarsa_plot_policy(goal, policy,lmbda)    
        
    sr.sarsa_avg_plot_all(r_avg_comp,s_avg_comp,episodes,goal,lambda_list)

    
