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


################################################################################################################################
# trained model
path = '/home/juntawu/my_work/gym_trocs/'
year = 2018
month = 4
day = 3
episode = 3450
# actor_model = path + 'trained_model_%d_%02d_%02d/actormodel-%d-%02d-%02d-episode%d.h5' % (year, month, day, year, month, day, episode)
# critic_model = path + 'trained_model_%d_%02d_%02d/criticmodel-%d-%02d-%02d-episode%d.h5' % (year, month, day, year, month, day, episode)
actor_model = path + 'trained_model_%d_%02d_%02d/actormodel_%d_%02d_%02d_episode%d.h5' % (year, month, day, year, month, day, episode)
critic_model = path + 'trained_model_%d_%02d_%02d/criticmodel_%d_%02d_%02d_episode%d.h5' % (year, month, day, year, month, day, episode)
################################################################################################################################


def playGame(train_indicator=0):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99


    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    action_dim = 3  # Steering/Acceleration/Brake
    state_dim = 29  # of sensors input

    np.random.seed(1337)  # Set seed of random module

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 2000
    reward = 0
    done = False
    step = 0

    epsilon = 1
    indicator = 0

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    # Generate a Torcs environment

    # env = TorcsEnv(vision=vision, throttle=True, gear_change=False, render = False)
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False, render=True)
    # env = TorcsEnv(vision=vision, throttle=True, gear_change=True)


    # Now load the weight
    print("Now we load the weight")
    actor.model.load_weights(actor_model)
    critic.model.load_weights(critic_model)
    actor.target_model.load_weights(actor_model)
    critic.target_model.load_weights(critic_model)
    print("Weight load successfully")


    print("TORCS Experiment Start.")
    for i in range(episode_count):
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        #if np.mod(i, 3) == 0:
        if i == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        #[WJT]: start of the car
        #for k in range(10):
        #    env.step(np.array([0, 1 ,0]))

        s_t = np.hstack(
            (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
	
        #[WJT]: testing the current model
        flag_pass_track = False
        max_test_steps = 5000
        if(0):
            print('Testing...')
            for test_step in range(max_test_steps):
                print('Step', test_step, 'Laps', env.client.S.laps)
                a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                ob, r_t, done, info, episode_terminate = env.step(a_t[0])
                if done:
                    break
                env.client.S.check_laps()
                if env.client.S.laps > 1:
                    flag_pass_track = True
                    print('The car pass the track')
                    #break
                s_t = np.hstack(
                    (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            ob, r_t, done, info, episode_terminate = env.step(a_t[0])

            s_t1 = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))


            total_reward += r_t
            s_t = s_t1

            #print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

            #[WJT] for the stability of connection
            if i == 0:
                break


        #[WJT]: testing the current model
        flag_pass_track = False
        max_test_steps = 5000
        if(1):
            print('Testing...')
            ob = env.reset()
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            env.test = True
            flag_pass_track = False
            max_test_steps = 5000
            for test_step in range(max_test_steps):
                print('Step', test_step, 'Laps', env.client.S.laps)
                a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                ob, r_t, done, info, episode_terminate = env.step(a_t[0])
                if done:
                    break
                env.client.S.check_laps()
                print('Step', test_step, 'Laps', env.client.S.laps)
                if env.client.S.laps > 1:
                    flag_pass_track = True
                    print('The car pass the track')
                    break
                s_t = np.hstack(
                    (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            env.test = False


        #print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))

        #print("Total Step: " + str(step))
        #print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")



if __name__ == "__main__":
    playGame()
