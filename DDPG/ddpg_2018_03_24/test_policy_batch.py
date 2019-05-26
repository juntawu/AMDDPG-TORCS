from gym_torcs import TorcsEnv
import numpy as np
import tensorflow as tf

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

import os



################################################################################################################################
# trained model
path = '/home/juntawu/my_work/gym_trocs/'
year = 2018
month = 4
day = 17
track = 'CG1'
#actor_model = path + 'trained_model_%d_%02d_%02d/actormodel_%d_%02d_%02d_episode%d.h5' % (year, month, day, year, month, day, episode)
#critic_model = path + 'trained_model_%d_%02d_%02d/criticmodel_%d_%02d_%02d_episode%d.h5' % (year, month, day, year, month, day, episode)
################################################################################################################################


def test():
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
    episode_count = 2
    reward = 0
    done = False
    step = 0

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
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False, render = False)
    # env = TorcsEnv(vision=vision, throttle=True, gear_change=False, render=True)
    # env = TorcsEnv(vision=vision, throttle=True, gear_change=True)

    all_files = sorted( os.listdir(path + 'trained_model_%d_%02d_%02d' % (year, month, day) ) )
    model_files = []
    for file in all_files:
        if(file[0:10] == 'actormodel'):
            model_files.append(file)

    test_result_file = open(path + 'trained_model_%d_%02d_%02d/' % (year, month, day) +'test_on_' + track + '.txt', 'w')
    test_result_file.write('models passing ' + track + ':\n')
    for file in model_files:
        #print(file)
        actor_model = path + 'trained_model_%d_%02d_%02d/' % (year, month, day) + file
        critic_model = path + 'trained_model_%d_%02d_%02d/' % (year, month, day) + 'criticmodel' + file[10:]

        # Now load the weight
        print("Now we load the weight")
        actor.model.load_weights(actor_model)
        critic.model.load_weights(critic_model)
        actor.target_model.load_weights(actor_model)
        critic.target_model.load_weights(critic_model)
        print("Weight load successfully")


        print("TORCS Experiment Start.")
        for i in range(2):
            print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
            if i == 0:
                ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
            else:
                ob = env.reset()

            s_t = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            ob, r_t, done, info, episode_terminate = env.step(a_t[0])
            # [WJT] for the stability of connection
            if i == 0:
                continue

            #[WJT]: testing the current model
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            env.test = True
            flag_pass_track = False
            max_test_steps = 5000
            print("Testing " + file + "......")
            for test_step in range(max_test_steps):
                a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                ob, r_t, done, info, episode_terminate = env.step(a_t[0])
                if done:
                    break
                env.client.S.check_laps()
                #print('Step', test_step, 'Laps', env.client.S.laps)
                if env.client.S.laps > 1:
                    flag_pass_track = True
                    #print('The car pass the track')
                    break
                s_t = np.hstack(
                    (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            env.test = False

        if(True == flag_pass_track):
            print('The car pass the ' + track)
            test_result_file.write(file + '\n')


    env.end()  # This is for shutting down TORCS
    print("Finish.")
    test_result_file.close()



if __name__ == "__main__":
    test()
