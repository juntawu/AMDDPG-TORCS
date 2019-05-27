from DDPG import DDPG
from ReplayBuffer import ReplayBuffer
import tensorflow as tf
import numpy as np

from gym_torcs import TorcsEnv
from OU import OU
import time
import os
import json

OU = OU()  # Ornstein-Uhlenbeck Process

################################################################################################################################
track_name = 'Aalborg'
################################################################################################################################
track_length_dict = {'CG1': 2957.56, 'CG2': 3185.83, 'CG3': 2843.10, 'Aalborg': 2587.54}


def trainModel():

    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    #GPU_USAGE = 1
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_USAGE
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    from keras import backend as K
    K.set_session(sess)

    #km = joblib.load('kmeans_model_2018_04_19/km.pkl')

    # create ddpg model
    ddpg_0 = DDPG(sess, 0)
    ddpg_1 = DDPG(sess, 1)
    ddpg_2 = DDPG(sess, 2)
    ddpg_0.load()
    ddpg_1.load()
    ddpg_2.load()
    """
    ddpg_0.load_pretrained_model(2018, 4, 3, 3450)
    ddpg_1.load_pretrained_model(2018, 4, 3, 3450)
    ddpg_2.load_pretrained_model(2018, 4, 3, 3450)
    """
    # Generate a Torcs environment
    #env = TorcsEnv(vision=False, throttle=True, gear_change=False, render=True)
    env = TorcsEnv(vision=False, throttle=True, gear_change=False, render=False)

    episode_count = 4000
    #max_steps = 100000
    max_steps = 5000
    EXPLORE = 100000.
    epsilon = 1
    reward = 0
    done = False
    step = 0

    state_dim = 29
    action_dim = 3

    #[WJT]: open file for saving information
    training_info = open("../trained_model/training_info.json", 'w')
    training_reward = open("../trained_model/training_reward.json", 'w')
    training_time = open("../trained_model/training_time.txt", 'w')
    

    print("TORCS Experiment Start.")
    init_time = time.time()
    for i in range(episode_count):

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

        episode_reward = 0.
        episode_step = 0
        
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            #cluster = int( km.predict(s_t.reshape(1, -1)) )
            # print(cluster)
            
            '''
            if 0==cluster:
                a_t_original = ddpg_0.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            elif 1==cluster:
                a_t_original = ddpg_1.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            elif 2==cluster:
                a_t_original = ddpg_2.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            '''
            if 0==np.mod(i,3):
                a_t_original = ddpg_0.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            elif 1==np.mod(i,3):
                a_t_original = ddpg_1.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            elif 2==np.mod(i,3):
                a_t_original = ddpg_2.actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            noise_t[0][0] = max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            noise_t[0][1] = max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            noise_t[0][2] = max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info, stop_flag = env.step(a_t[0], track_name)

            s_t1 = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

            """
            ## training after clustering
            if 0 == cluster:
                ddpg_0.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                ddpg_0.train()
            if 1 == cluster:
                ddpg_1.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                ddpg_1.train()
            if 2 == cluster:
                ddpg_2.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                ddpg_2.train()
            """
            
            """
            ## training by order using different experience buffer
            if 0 == np.mod(j,3):
                ddpg_0.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                ddpg_0.train()
            if 1 == np.mod(j,3):
                ddpg_1.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                ddpg_1.train()
            if 2 == np.mod(j,3):
                ddpg_2.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                ddpg_2.train()
            """
            
            
            ## training by order using same experience buffer
            """
            # filter experiences
            if step <= EXPLORE:
                ddpg_0.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                ddpg_1.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                ddpg_2.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
            else:
                if r_t > 10:
                    ddpg_0.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                    ddpg_1.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
                    ddpg_2.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
            """
            
            ddpg_0.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
            ddpg_1.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
            ddpg_2.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
            
            if 0 == np.mod(i,3):
                ddpg_0.train()
            if 1 == np.mod(i,3):
                ddpg_1.train()
            if 2 == np.mod(i,3):
                ddpg_2.train()
            
            episode_reward += r_t
            episode_step += 1
            s_t = s_t1

            # print training info
            #print("Episodes", i, "Steps", step, "Cluster", cluster, "Reward", r_t)
            print("Episodes", i, "Steps", step,  "Reward", r_t)
            # save training info
            #training_info.write(str(i) + ';' + str(step) + ';' + str(r_t) + ';' + str(cluster) + ';' + '\n')
            #training_info.write(str(i) + ';' + str(step) + ';' + str(r_t) + ';' + '\n')
            json.dump({"Episode": i, "Total Steps": step, "Step Reward": r_t}, training_info)
            training_info.write("\n")

            step += 1
            if done:
                break

            # [WJT] for the stability of connection
            if i == 0:
                break

        ##############################################################################################################
        #[wjt]: save model when the car can pass the track with this model
    	# if(i>=1000 and (i+1)%100==0):
    	#if( (i < 2000 and (i+1) % 100 == 0) or (i > 2000 and (i+1) % 50 == 0) ):
    	if((i+1)%50==0):
            print ('Testing the model ......')
            ob = env.reset()
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
               
            #env.test = True
            flag_pass_track = False
            max_test_steps = 5000
            for test_step in range(max_test_steps):
                #a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                #cluster = int( km.predict(s_t.reshape(1, -1)) )
                # print(cluster)
                '''
                if 0==cluster:
                    a_t = ddpg_0.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                elif 1==cluster:
                    a_t = ddpg_1.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                elif 2==cluster:
                    a_t = ddpg_2.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                '''
           
                a_t_0 = ddpg_0.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                a_t_1 = ddpg_1.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                a_t_2 = ddpg_2.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                a_t[0] = np.mean([a_t_0[0], a_t_1[0], a_t_2[0]], axis=0)
                """
                print("a_t_0: ", a_t_0[0])
                print("a_t_1: ", a_t_1[0])
                print("a_t_2: ", a_t_2[0])
                print("a_t: ", a_t[0])
                """
                ob, r_t, done, info, episode_terminate = env.step(a_t[0], track_name)
                
                print("Testing Steps", test_step,  "Reward", r_t)
               
                if done:
                    break
                env.client.S.check_laps()
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
                print("The car pass the track! Now we save model")
                training_time.write('trianing time of episode%d.h5: ' %(i+1) + str( (time.time()-init_time)/3600 ) + 'hours\n')
                # [WJT]: save model when
                ddpg_0.save_trained_model(i + 1)
                ddpg_1.save_trained_model(i + 1)
                ddpg_2.save_trained_model(i + 1)
        env.test = False
    ##############################################################################################################
   
        json.dump({"Episode": i, "Episode Reward": episode_reward, "Episode Step": episode_step, "Mean Reward": episode_reward/float(episode_step)}, training_reward)
        training_reward.write("\n")
        print("Episode REWARD @ " + str(i) + "-th Episode  : Reward " + str(episode_reward))
        print("Episode Step: " + str(episode_step))
        print("")


    # save experiences
    #ddpg_0.save_experiences()
    #ddpg_1.save_experiences()
    #ddpg_2.save_experiences()
    
    env.end() # this is for shutting down TORCS
    print("Finish.")
    
    #[WJT]: close file
    training_info.close()
    training_reward.close()
    training_time.close()


if __name__ == "__main__":
    start_time = time.time()
    trainModel()
    end_time = time.time()
    total_hours = (end_time - start_time) / 3600
    print("Total training time: %f hours \n" % total_hours)
