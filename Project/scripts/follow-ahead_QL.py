from rl_trainer import RL_Trainer
from dqn_agent import DQNAgent
from navi_state import navState
from replayBuffer import replay_buffer
import numpy as np



class Q_Trainer(object):

    def __init__(self, params):
        self.params = params

        robot_angle_acts = np.arange(-self.params["action_size"], self.params["action_size"]+1) * self.params["action_step"] * np.pi /180
        robot_vel_acts = np.array([0.1, 0.2])
        self.params['ac_dim'] = (2* self.params['action_size'] + 1) * len(robot_vel_acts)
        human_actions = np.array([-10, 10])* np.pi /180 
        
        ################ defining wave trajectory
# =============================================================================
#         human_x = robot_vel_acts[0] * np.arange(1, self.params['ep_length']+1)
#         human_y = 0.5 * np.cos(human_x * np.pi / 2 + np.pi ) + 0.5
#         human_theta = [np.arctan2(human_y[i+1] - human_y[i], human_x[i+1] - human_x[i]) for i in range(len(human_x)-1)]           
#         human_theta.append(human_theta[len(human_theta)-1]) 
#         
#         human_actions = [human_x, human_y, human_theta]
# =============================================================================
        
        self.params['robot_vel_acts'] = robot_vel_acts
        self.initial_state= np.array([[1.5, 0, 0],[0, 0, 0]])
        self.params['sample_gen_size'] = 5

        self.agent = DQNAgent(params)
        self.params['agent_class'] = self.agent
        self.params['robot_angle_acts'] = robot_angle_acts
        self.params['human_actions'] = human_actions
        self.params['initial_state'] = self.initial_state
        self.params['exp_name'] = 'n:'+str(params['n_iterations'])+ '-nLayers:'+str(params['n_layer'])+ '-lsize:'+str(params['layer_size'])+ '-bs:'+str(params['batch_size'])+ '-lr:'+ str(params['learning_rate']) + '-ep:'+ str(params['ep_length'])+ '-test:'+ str(params['test']) + '-tar_up'+ str(params['target_update_freq']) + '-aSize' + str(params['action_size']) + '-aStep' + str(params['action_step'])+'-2vel'
        self.params['save_dir'] = '/home/sahar/Follow-ahead/results_QN/'
        self.params['init_state'] = self.initial_state
        self.params['max_reward_per_ep'] = 1
        self.params['navstate'] = navState
        

        self.BF = replay_buffer(params= params)
        self.params['buffer'] = self.BF

        if self.params['generate_random_sample']:
            self.generate_random_samples(robot_angle_acts, human_actions, 100000, self.params['ep_length'], self.initial_state)

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            policy = self.agent.actor,
        )

    def generate_random_samples(self, robot_angle_acts, human_actions, sample_size, ep_length, init_state):
        
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
            
            #human_move = [human_actions[0]] if s<sample_size/2 else [human_actions[1]]
            
            
            for ep in range(ep_length):
                #human_move = np.random.choice(human_actions)
                #human_move = human_actions
                human_move = [self.params['human_actions'][0][ep], self.params['human_actions'][1][ep] , self.params['human_actions'][2][ep]]
                state, _ = nav_state.calculate_new_state(state, human_move, 0.1, next_to_move = -1)
                    
                robot_move = np.random.choice(robot_angle_acts)
                new_state, reward = nav_state.calculate_new_state(state, robot_move, vel , next_to_move = 1)
                self.BF.add_sample(state, np.where(robot_angle_acts == robot_move)[0][0], v_ind, new_state, reward, 0)

                state = new_state

        self.BF.save_to_file()


def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--test', '-t', type=int, default=1)
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--target_update_freq', '-tuf', type=int, default=30)
    parser.add_argument('--action_size', type=int, default= 1)
    parser.add_argument('--action_step', type=float, default= 20.)
    parser.add_argument('--n_layer', '-nl', type=int, default= 1)
    parser.add_argument('--layer_size', '-ls', type=int, default= 16)
    parser.add_argument('--learning_rate', '-lr', type=float, default= 1e-2)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--n_iterations', '-n', type=int, default= 150)
    parser.add_argument('--ep_length', '-ep', type=int, default= 15)
    parser.add_argument('--test_size', type=int, default= 100)
    parser.add_argument('--generate_random_sample', type=int, default= 0)
    parser.add_argument('--use_dataset', type=int, default= 1)
    parser.add_argument('--dataset_name', type=str, default= 'left-right-10D_dis-R')
    
    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    trainer = Q_Trainer(params)
    if not params['generate_random_sample']:
        trainer.run_training_loop()


if __name__ == "__main__":
    main()
    print('Done!')