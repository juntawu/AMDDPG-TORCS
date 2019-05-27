from sklearn.cluster import KMeans
from sklearn.externals import joblib
from DDPG import DDPG
from ReplayBuffer import ReplayBuffer
import tensorflow as tf
import numpy as np

from gym_torcs import TorcsEnv
from OU import OU
import time
import os

import matplotlib.pyplot as plt
import pdb
import json


OU = OU()  # Ornstein-Uhlenbeck Process


################################################################################################################################
# trained model
year = 2018
month = 6
day = 10
episode = 5900
track_name = 'Aalborg'
################################################################################################################################

track_length_dict = {'CG1': 2057.56, 'CG2': 3185.83, 'CG3': 2843.10, 'Aalborg': 2587.54}




def testModel():

    TrackPos = []
    Speed = []
    Angle = []
    Reward = []
    

    #os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    #GPU_USAGE = 0.5
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_USAGE
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    from keras import backend as K
    K.set_session(sess)


    # create ddpg model
    ddpg_0 = DDPG(sess, 0)
    ddpg_1 = DDPG(sess, 1)
    ddpg_2 = DDPG(sess, 2)
    
    ddpg_0.load_trained_model(year, month, day, episode)
    ddpg_1.load_trained_model(year, month, day, episode)
    ddpg_2.load_trained_model(year, month, day, episode)
    '''
    ddpg_0.load_pretrained_model(year, month, day, episode)
    ddpg_1.load_pretrained_model(year, month, day, episode)
    ddpg_2.load_pretrained_model(year, month, day, episode)
    '''

    # Generate a Torcs environment
    #env = TorcsEnv(vision=False, throttle=True, gear_change=False, render=True)
    env = TorcsEnv(vision=False, throttle=True, gear_change=False, render=False)

    episode_count = 6
    #max_steps = 100000
    max_steps = 5000
    EXPLORE = 100000.
    epsilon = 1
    reward = 0
    done = False
    step = 0

    state_dim = 29
    action_dim = 3

    #action_file = open('trained_model_%d_%02d_%02d/result_analysis/' %(year, month, day) + 'action_%d_episode_%s.json' %(episode, track_name), 'w')
    print("TORCS Experiment Start.")
    for i in range(episode_count):
        
        episode_trackPos = []
        episode_speed = []
        episode_angle = []
        episode_reward = []
        
        if i == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            

        ##############################################################################################################
        print ('Testing the model ......')
        
        #env.test = True
        flag_pass_track = False
        max_test_steps = 5000
        total_reward = 0
        for test_step in range(max_test_steps):
            
            a_t_0 = ddpg_0.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            a_t_1 = ddpg_1.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            a_t_2 = ddpg_2.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            a_t = np.mean([a_t_0[0], a_t_1[0], a_t_2[0]], axis=0)
            
            #ob, r_t, done, info, episode_terminate = env.step(a_t, track_name)
            if(i == 2):
                action = a_t_0[0]
            elif(i == 3):
                action = a_t_1[0]
            elif(i == 4):
                action = a_t_2[0]
            else:
                action = a_t
            ob, r_t, done, info, episode_terminate = env.step(action, track_name)
            
            """
            print("a_t_0: ", a_t_0[0])
            print("a_t_1: ", a_t_1[0])
            print("a_t_2: ", a_t_2[0])
            print("a_t: ", a_t)
            """
            #a_t = ddpg_0.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            #ob, r_t, done, info, episode_terminate = env.step(a_t[0])
            
            
            episode_trackPos.append(ob.trackPos)
            episode_speed.append(ob.speedX * 50)
            episode_angle.append(ob.angle)
            total_reward += r_t
            episode_reward.append(total_reward)
            
            if done:
                if i>=2:
                    print(list(action))
                    action = [float(list(action)[i]) for i in range(3)]
                    #json.dump(action, action_file) 
                break
            env.client.S.check_laps()
            #print('cluster: ' + str(cluster) + ';', 'laps: ' + str(env.client.S.laps))
            if env.client.S.laps == 0:
                    dist_from_start = 0
            else:
                dist_from_start = env.client.S.read_distFromStart()
            if dist_from_start >= track_length_dict[track_name]-10:
                flag_pass_track = True
                if i>=2:
                    print(list(action))
                    action = [float(list(action)[i]) for i in range(3)]
                    #json.dump(action, action_file) 
                break
            s_t = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            

        if (flag_pass_track):
            print("Episode %d: The car pass the track" % i)
               
        env.test = False
        
        if(i>=2):
            TrackPos.append(episode_trackPos)
            Speed.append(episode_speed)
            Angle.append(episode_angle)
            Reward.append(episode_reward)
        ##############################################################################################################


    
    env.end() # this is for shutting down TORCS
    print("Finish.")
    
    ## save the reward into json file
    reward_file = open('trained_model_%d_%02d_%02d/result_analysis/' %(year, month, day) + 'reward_%d_episode_%s.json' %(episode, track_name), 'w')
    json.dump(Reward, reward_file)
    
    #return [TrackPos, Speed, Angle, Reward]
    return [Reward]


def plot_performance(info ):
    
    #y_label = ['trackpos', 'speed', 'angle', 'reward']
    y_label = ['reward'] 
    #pdb.set_trace()
    for i in range(len(info)):
        plt.figure(i)
        plt.plot(info[i][0], color='blue', label='DDPG0')
        plt.plot(info[i][1], color='green', label='DDPG1')
        plt.plot(info[i][2], color='purple', label='DDPG2')
        plt.plot(info[i][3], color='red', label='MDDPGs')
        plt.xlabel('Step')
        plt.ylabel(y_label[i])
        plt.legend()
        plt.savefig('trained_model_%d_%02d_%02d/result_analysis/' %(year, month, day) + '%depisode_%s.png' % (episode, y_label[i]))
    
    
    


if __name__ == "__main__":
    info = testModel()
    #plot_performance(info)
    
