from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse


from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import time
import os

################################################################################################################################
# usage of gpu
GPU_USAGE = 1

# pre-trained model
path = '/home/wjt/my_work/gym_torcs/'
year = 2017
month = 12
day = 29
episode = 3199
actor_model = path + 'model_origin/actormodel.h5'
critic_model = path + 'model_origin/criticmodel.h5'
# actor_model = path + 'trained_model_%d_%02d_%02d/actormodel-%d-%02d-%02d-episode%d.h5' % (year, month, day, year, month, day, episode)
# critic_model = path + 'trained_model_%d_%02d_%02d/criticmodel-%d-%02d-%02d-episode%d.h5' % (year, month, day, year, month, day, episode)

# path for saving model
save_path = path + 'trained_model/'
################################################################################################################################



ISOTIMEFORMAT='%Y_%m_%d'
time_stamp = time.strftime(ISOTIMEFORMAT, time.gmtime(time.time()))

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    np.random.seed(1337)  #Set seed of random module

    vision = False

    EXPLORE = 100000.
    episode_count = 4000
    max_steps = 2000
    reward = 0
    done = False
    step = 0


    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    #os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5,6,7"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_USAGE
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False, render = False)
    # env = TorcsEnv(vision=vision, throttle=True, gear_change=True)

    #Now load the weight
    actor.model.load_weights(actor_model)
    critic.model.load_weights(critic_model)
    actor.target_model.load_weights(actor_model)
    critic.target_model.load_weights(critic_model)
    print("Weight load successfully")


    #[WJT]: open file for saving information
    training_info = open(save_path + "training_info.txt", 'w')
    experience_replay = open(save_path + "experience_replay.txt", 'w')

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()


        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
     
        total_reward = 0.

        # [WJT]: steps of current episode
        steps_episode = 0
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info, episode_terminate = env.step(a_t[0])

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]  #???
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                # part of policy gradient: derivative of Q w.r.t. a
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
            print(buff.num_experiences)
            # training_info.write(str(i) + ';' + str(step) + ';' + str(r_t) + ';' + str(loss) + ';' + '\n')

            steps_episode += 1
            step += 1
            if done:
                break

            # [WJT] for the stability of connection
            if i == 0:
                break


    ##############################################################################################################
        #[wjt]: save model when the car can pass the track with this model
	# if(i>=1000 and (i+1)%100==0):
	if( (i+1) % 100 == 0 ):
        print ('Testing the model ......')
        ob = env.reset()
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
        env.test = True
        flag_pass_track = False
        max_test_steps = 5000
        for test_step in range(max_test_steps):
                a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                ob, r_t, done, info, episode_terminate = env.step(a_t[0])
                if done:
                    break
                env.client.S.check_laps()
                if env.client.S.laps > 1:
                    flag_pass_track = True
                    break
                s_t = np.hstack(
                    (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

            if (flag_pass_track):
                print("Now we save model")

                actor.model.save_weights(save_path + "actormodel_"+time_stamp+"_episode%d.h5" %(i+1), overwrite=True)
                # with open("trained_model/actormodel-"+time_stamp+"-episode%d.json" %i, "w") as outfile:
                #     json.dump(actor.model.to_json(), outfile)


                critic.model.save_weights(save_path + "criticmodel_"+time_stamp+"_episode%d.h5" %(i+1), overwrite=True)
                # with open("trained_model/criticmodel-"+time_stamp+"-episode%d.json" %i, "w") as outfile:
                #     json.dump(critic.model.to_json(), outfile)
	    env.test = False
    ##############################################################################################################


        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))

        print("Total Step: " + str(step))
        print("")

    #[WJT]: save experiences
    #for k in range(buff.num_experiences):
        #for l in range(29):  # save s_t
            #experience_replay.write(str(buff.buffer[k][0][l]) + ';')
        #for l in range(3):
            #experience_replay.write(str(buff.buffer[k][1][l]) + ';')  # a_t
        #experience_replay.write(str(buff.buffer[k][2]) + ';')  # r_t
        #for l in range(29):  # s_t1
            #experience_replay.write(str(buff.buffer[k][3][l]) + ';')
        #experience_replay.write(str(buff.buffer[k][4]))  # done
        #experience_replay.write('\n')


    env.end()  # This is for shutting down TORCS
    print("Finish.")

    #[WJT]: close file
    training_info.close()
    experience_replay.close()


if __name__ == "__main__":
    start_time = time.time()
    playGame()
    end_time = time.time()
    total_hours = (end_time - start_time) / 3600
    print("Total training time: %f hours \n" % total_hours)
