import numpy as np
import matplotlib.pyplot as plt 
from navi_state import navState
import pickle

class RL_Trainer(object):

    def __init__(self, params):
        self.params = params
        self.agent = self.params['agent_class']
        self.BF = self.params['buffer']
        self.test_reward_max = self.params['ep_length'] * self.params['max_reward_per_ep'] * -1
        self.eval_traj = []


    def run_training_loop(self, policy):
        all_logs = []
        test_mean = []
        test_std = []
        
        if self.params['use_dataset']:
            print("Loading dataset")
            with open("/home/sahar/Follow-ahead/dataset/"+self.params['dataset_name'], "rb") as fp:   # Unpickling
                dataset = pickle.load(fp)
        
        for itr in range(self.params['n_iterations']):
                
            if itr % 10 == 0:
                print("----iteration: "+ str(itr))

            if self.params['use_dataset']:
                rand_indices = np.random.permutation(len(dataset))[:self.params['batch_size']]
                rollouts = [dataset[r] for r in rand_indices]
                ob_b, angle_b, vel_b, next_ob_b, r_b, t_b = self.BF.convert_rollouts_to_list(rollouts)
            else:
                if itr % (self.params['n_iterations']/5) == 0:
                    self.BF.generate_sample(self.params['sample_gen_size'], itr) # times greater than batch size
                    ob_b, angle_b, vel_b, next_ob_b, r_b, t_b = self.BF.sample_random_data(self.params['batch_size'])


            ac_b = angle_b + len(self.params['robot_angle_acts']) * vel_b
            for _ in range(self.params['num_critic_updates_per_agent_update']):
                train_log = self.agent.train(ob_b, ac_b, r_b, next_ob_b, t_b)
                all_logs.append(train_log)
                
            mean, std = self.test()
            test_mean.append(mean)
            test_std.append(std)
            
            if mean > self.test_reward_max: 
                self.agent.save_net()
                self.evaluation()
                self.test_reward_max = mean
             
        if len(self.eval_traj): 
            self.plot_eval_traj()
            
        self.plotting(all_logs, test_mean, test_std)        
        return all_logs        

    def test(self):
        R = []
        for h in range(len(self.params['human_actions'])):
            for _ in range(self.params['test_size']):
                state = self.params['init_state']
                nav_state = navState(state)
                ep_length = self.params['ep_length']
                sum_of_rewards = 0
            
                for ep in range(ep_length):
                    human_move = [self.params['human_actions'][h]]
                    #human_move = [self.params['human_actions'][0][ep], self.params['human_actions'][1][ep] , self.params['human_actions'][2][ep]]
                    state, _ = nav_state.calculate_new_state(state, human_move, 0.1, next_to_move = -1)
    
                    obs = state[1,:] - state[0,:]
                    
                    actions = self.agent.forward(obs)
                    vel_ind = 1
                    while actions >= len(self.params['robot_angle_acts']):
                        actions -= len(self.params['robot_angle_acts'])
                        vel_ind +=1
        
                    robot_move = self.params['robot_angle_acts'][actions]
                    new_state, reward = nav_state.calculate_new_state(state, robot_move, 0.1 * vel_ind , next_to_move = 1)
                    sum_of_rewards += reward
        
                    state = new_state
    
                R.append(sum_of_rewards)
            
        return np.mean(R), np.std(R)
    
    def evaluation(self):  
        all_traj = []
        for h in range(len(self.params['human_actions'])):
            traj = []
            ep_length = self.params['ep_length']
            state = self.params['initial_state']
            traj.append(state)
            nav_state = navState(state)
        
            for ep in range(ep_length):
                human_move = [self.params['human_actions'][h]]
                #human_move = [self.params['human_actions'][0][ep], self.params['human_actions'][1][ep] , self.params['human_actions'][2][ep]]
                state, _ = nav_state.calculate_new_state(state, human_move, 0.1, next_to_move = -1)
                
                obs = state[1,:] - state[0,:]
                actions = self.agent.forward(obs)
                vel_ind = 1
                while actions >= len(self.params['robot_angle_acts']):
                    actions -= len(self.params['robot_angle_acts'])
                    vel_ind +=1
    
                robot_move = self.params['robot_angle_acts'][actions]
                new_state, reward = nav_state.calculate_new_state(state, robot_move, 0.1 * vel_ind , next_to_move = 1)
    
                state = new_state
                traj.append(state)
                
            all_traj.append(traj)
            
        self.eval_traj = all_traj
            
            
    def plot_eval_traj(self): 
        
        color = ['blue', 'deepskyblue', 'red', 'lightcoral', 'green', 'lightgreen']
        
        for h in range(len(self.eval_traj)):           
            robot_x = [xr[0][0] for xr in self.eval_traj[h]]
            robot_y = [yr[0][1] for yr in self.eval_traj[h]]
            human_x = [xh[1][0] for xh in self.eval_traj[h]]
            human_y = [yh[1][1] for yh in self.eval_traj[h]]
    
            plt.figure()
            plt.plot(robot_x, robot_y, c = color[2*h], label='Robot', marker='.')
            plt.plot(human_x, human_y, c = color[2*h +1], label = 'Human', marker='.')
        
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            #plt.ylim([-2,2])
            plt.axis('equal')
            plt.legend()
            plt.savefig(self.params['save_dir']+ self.params['exp_name']+'-traj-'+ str(h) +'.png')

# =============================================================================
#             plt.figure()
#             plt.plot(robot_x, robot_y, c = 'blue', label='Robot', marker='.')
#             plt.plot(human_x, human_y, c = 'red', label = 'Human', marker='.')
#             plt.xlabel('X')
#             plt.ylabel('Y')
#             plt.axis('equal')
#             plt.legend()
#             plt.savefig(self.params['save_dir']+ self.params['exp_name']+'-traj-equal-' + str(h) +'.png')
# =============================================================================
                
    def plotting(self, logs, test_mean, test_std):
        
        train_loss = np.array([l['Training Loss']  for l in logs])
        iteration = np.arange(len(train_loss))+1
        plt.figure()
        plt.plot(iteration, train_loss)
        plt.xlabel('iterations')
        plt.ylabel('Training Loss')
        plt.savefig(self.params['save_dir']+ self.params['exp_name']+'-loss.png')
 
        iteration = np.arange(len(test_mean))+1
        plt.figure()
        plt.errorbar(iteration, test_mean, test_std, ecolor='lightgray')
        plt.xlabel('Iterations')
        plt.ylabel('Test reward')
        plt.savefig(self.params['save_dir']+ self.params['exp_name']+'-test_reward.png')  

    def generate_sample(self, robot_angle_acts, human_actions, sample_size, ep_length, init_state):
        
        ss = int(sample_size / len(self.params['robot_vel_acts']))
        vel_list = np.zeros(sample_size)
        v_ind_list = np.zeros(sample_size)
        
        for i in range(len(self.params['robot_vel_acts'])):
            v_ind_list[i*ss:(i+1)*ss] = i
            vel_list[i*ss:(i+1)*ss] = self.params['robot_vel_acts'][i] 
        
            
        for s in range(sample_size):
            if s % 1000 == 0:
                print(s)
                
            state = init_state
            nav_state = navState(state)
            vel = vel_list[s]
            v_ind = v_ind_list[s]
            
            for ep in range(ep_length):
                #human_move = np.random.choice(human_actions)
                human_move = [self.params['human_actions'][0][ep], self.params['human_actions'][1][ep] , self.params['human_actions'][2][ep]]
                state, _ = nav_state.calculate_new_state(state, human_move, 0.1, next_to_move = -1)
                    
                robot_move = np.random.choice(robot_angle_acts)
                new_state, reward = nav_state.calculate_new_state(state, robot_move, vel , next_to_move = 1)
                self.BF.add_sample(state, np.where(robot_angle_acts == robot_move)[0][0], v_ind, new_state, reward, 0)

                state = new_state
                
                
                
                
                