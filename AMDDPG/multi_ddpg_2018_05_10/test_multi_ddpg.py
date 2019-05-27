from DDPG import DDPG
from ReplayBuffer import ReplayBuffer
import tensorflow as tf
import numpy as np

from gym_torcs import TorcsEnv
from OU import OU
import time
import os
import matplotlib.pyplot as plt
import json

OU = OU()  # Ornstein-Uhlenbeck Process


################################################################################################################################
# trained model
year = 2018
month = 5
day = 16
episode = 5500
track_name = 'Aalborg'
################################################################################################################################

track_length_dict = {'CG1': 2057.56, 'CG2': 3185.83, 'CG3': 2843.10, 'Aalborg': 2587.54}

def testModel():

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
    # set render to be "True" to observe the car's behaviour
    env = TorcsEnv(vision=False, throttle=True, gear_change=False, render=True)
    #env = TorcsEnv(vision=False, throttle=True, gear_change=False, render=False)

    episode_count = 3
    #max_steps = 100000
    max_steps = 5000
    EXPLORE = 100000.
    epsilon = 1
    reward = 0
    done = False
    step = 0

    state_dim = 29
    action_dim = 3



    print("TORCS Experiment Start.")
    for i in range(episode_count):
    
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
        for test_step in range(max_test_steps):
            #cluster = int( km.predict(s_t.reshape(1, -1)) )
            #cluster = int( km.predict(ob.track.reshape(1, -1)) )
            '''
            left_track = np.mean(ob.track[0:9]) * 200
            right_track = np.mean(ob.track[10:-1]) * 200
            print([left_track, right_track])
            if(right_track - left_track > 10):
                cluster = 'turn right'
            elif(left_track - right_track > 10):
                cluster = 'turn left'
            else:
                cluster = 'straight'
            print('cluster: ' + cluster)
            '''
            """
            if 0==cluster:
                a_t = ddpg_0.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            elif 1==cluster:
                a_t = ddpg_1.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            elif 2==cluster:
                a_t = ddpg_2.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            """
            a_t_0 = ddpg_0.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            a_t_1 = ddpg_1.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            a_t_2 = ddpg_2.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            a_t = np.mean([a_t_0[0], a_t_1[0], a_t_2[0]], axis=0)
            ob, r_t, done, info, episode_terminate = env.step(a_t, track_name)
            #ob, r_t, done, info, episode_terminate = env.step(a_t_1[0], track_name)
            """
            print("a_t_0: ", a_t_0[0])
            print("a_t_1: ", a_t_1[0])
            print("a_t_2: ", a_t_2[0])
            print("a_t: ", a_t)
            """
            #a_t = ddpg_0.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            #ob, r_t, done, info, episode_terminate = env.step(a_t[0])
            
            print('step: ' + str(test_step), 'reward: ' + str(r_t))
            if 2 == i:
               episode_reward.append(r_t)
               
            
            if done:
                break
            env.client.S.check_laps()
            #print('cluster: ' + str(cluster) + ';', 'laps: ' + str(env.client.S.laps))
            if env.client.S.laps == 0:
                    dist_from_start = 0
            else:
                dist_from_start = env.client.S.read_distFromStart()
            if dist_from_start >= track_length_dict[track_name]-10:
                flag_pass_track = True
                break
            s_t = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            

        if (flag_pass_track):
            print("The car pass the track")
                
        env.test = False
        ##############################################################################################################


    
    env.end() # this is for shutting down TORCS
    print("Finish.")
    
    #reward_file = open('multi_%d_%02d_%02d_%depisode.json'%(year, month, day, episode), 'w')
    #json.dump(episode_reward, reward_file)
    #plt.plot(episode_reward)
    #plt.show()
    print('Total reward: ', np.sum(np.array(episode_reward)))



if __name__ == "__main__":
    testModel()
