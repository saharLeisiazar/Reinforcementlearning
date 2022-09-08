import numpy as np
import math 
import util as ptu


class navState(object):

    def __init__(self, state, next_to_move=1):
        self.state = state 
        self.reward = 0
        self.state_size = self.state.shape
        self.next_to_move = next_to_move


    def nextToMove(self):
        return self.next_to_move

    def move(self, move, dist):
        new_state, reward = self.calculate_new_state(self.state, move, dist)
        # if self.next_to_move == 1:
        #     ptu.add_sample_to_buffer(self.state, move, new_state, reward, 0) 
        next_to_move = -1 if self.next_to_move == 1 else 1
        return navState(new_state, next_to_move) 

    def calculate_new_state(self, state, action, dist, next_to_move = None ):
        if next_to_move == -1 and len(action) == 3:
            new_s = np.copy(state)
            new_s[1,0] = action[0]
            new_s[1,1] = action[1]
            new_s[1,2] = action[2]
            
            reward = None
        else:    
            if next_to_move == None:
                next_to_move = self.next_to_move
    
            ind = 0 if next_to_move == 1 else 1  # zero for moving robot, one for moving human
         
            new_s = np.copy(state)
            new_s[ind,0] = state[ind,0] + dist * math.cos(action + state[ind,2])
            new_s[ind,1] = state[ind,1] + dist * math.sin(action + state[ind,2])
            new_s[ind,2] = action + state[ind,2]
            reward = self.calculate_reward(new_s) if next_to_move == 1 else None

        return new_s, reward

    def calculate_reward(self, s):
        D = np.linalg.norm(s[0,:2]- s[1, :2]) #distance_to_human

        beta = math.atan2(s[0,1] - s[1,1] , s[0,0] - s[1,0])   # atan2 (yr - yh  , xr - xh)
        if (s[0,0] - s[1,0]) < 0:
            beta = beta + math.pi
            
        alpha = np.absolute(s[1,2] - beta)  #angle between the person-robot vector and the person-heading vector  
        alpha = alpha*180/np.pi
        
        theta = np.absolute((s[0,2] - s[1,2]) * 180 /np.pi)
   
        if (D <= 1 and D > 0.5):          
            Rd = -(1-D)
        elif (D <= 2 and D > 1):          
            Rd = 0.5*(0.5-np.absolute(D-1.5)) 
            #Rd = 0.25
        elif (D <= 5 and D > 2):          
            Rd = -0.25*(D-1)   
        else:
            Rd = -1 
               
        
        if alpha < 10:
            Ro = 0.5* ((10 - alpha)/10)
        else:
            Ro = -0.25 * alpha /180
        #Ro = -0.04 * alpha + 1
        
        Rt = (-1*theta + 10)*0.25 if theta < 10 else -1 
        Rt = Rt * 3 /180 

        return min(max(Rd, -1) , 1) #min(max(Ro + Rd + Rt, -1) , 1) min(max(Rd, -1) , 1)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  