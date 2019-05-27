from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import numpy as np
import time

BUFFER_SIZE = 100000
BATCH_SIZE = 32
TAU = 0.001
LRA = 0.0001    #Learning rate for Actor
LRC = 0.001     #Lerning rate for Critic
GAMMA = 0.99

state_dim = 29
action_dim = 3


ISOTIMEFORMAT='%Y_%m_%d'
time_stamp = time.strftime(ISOTIMEFORMAT, time.gmtime(time.time()))


class DDPG(object):

    def __init__(self, sess, num):
        self.actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
        self.critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
        self.buff = ReplayBuffer(BUFFER_SIZE)
        self.num = num


    def train(self):
        # Do the batch update
        batch = self.buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA * target_q_values[k]

        self.critic.model.train_on_batch([states, actions], y_t)
        a_for_grad = self.actor.model.predict(states)
        # part of policy gradient: derivative of Q w.r.t. a
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()


    def load(self):
        # Now load the weight
        """
        self.actor.model.load_weights('ddpg_model/actormodel_%d.h5' % self.num)
        self.critic.model.load_weights('ddpg_model/criticmodel_%d.h5' % self.num)
        self.actor.target_model.load_weights('ddpg_model/actormodel_%d.h5' % self.num)
        self.critic.target_model.load_weights('ddpg_model/criticmodel_%d.h5' % self.num)
        """
        self.actor.model.load_weights('../model_origin/actormodel.h5')
        self.critic.model.load_weights('../model_origin/criticmodel.h5')
        self.actor.target_model.load_weights('../model_origin/actormodel.h5')
        self.critic.target_model.load_weights('../model_origin/criticmodel.h5')
        print("Weight load successfully")

    def load_trained_model(self, year, month, day, episode):
        self.actor.model.load_weights('../trained_model_%d_%02d_%02d/actormodel_%d_%d_%02d_%02d_episode%d.h5' % (year, month, day, self.num, year, month, day, episode) )
        self.critic.model.load_weights('../trained_model_%d_%02d_%02d/criticmodel_%d_%d_%02d_%02d_episode%d.h5' % (year, month, day, self.num, year, month, day, episode))
        self.actor.target_model.load_weights('../trained_model_%d_%02d_%02d/actormodel_%d_%d_%02d_%02d_episode%d.h5' % (year, month, day, self.num, year, month, day, episode))
        self.critic.target_model.load_weights('../trained_model_%d_%02d_%02d/criticmodel_%d_%d_%02d_%02d_episode%d.h5' % (year, month, day, self.num, year, month, day, episode))
        print("Weight load successfully")

    def load_pretrained_model(self, year, month, day, episode):
        self.actor.model.load_weights('../trained_model_%d_%02d_%02d/actormodel_%d_%02d_%02d_episode%d.h5' % (year, month, day, year, month, day, episode) )
        self.critic.model.load_weights('../trained_model_%d_%02d_%02d/criticmodel_%d_%02d_%02d_episode%d.h5' % (year, month, day, year, month, day, episode))
        self.actor.target_model.load_weights('../trained_model_%d_%02d_%02d/actormodel_%d_%02d_%02d_episode%d.h5' % (year, month, day, year, month, day, episode))
        self.critic.target_model.load_weights('../trained_model_%d_%02d_%02d/criticmodel_%d_%02d_%02d_episode%d.h5' % (year, month, day, year, month, day, episode))
        print("Weight load successfully")



    def save(self):
        # save ddpg model
        self.actor.model.save_weights("ddpg_model/actormodel_" + str(self.num) + ".h5", overwrite=True)
        self.critic.model.save_weights("ddpg_model/criticmodel_" + str(self.num) + ".h5", overwrite=True)

    def save_trained_model(self, episodes):
        # print("trained_model/actormodel_%d_"%self.num + time_stamp+ "_episode%d.h5" % episodes)
        self.actor.model.save_weights("trained_model/actormodel_%d_"%self.num + time_stamp+ "_episode%d.h5" %episodes, overwrite=True)
        self.critic.model.save_weights("trained_model/criticmodel_%d_"%self.num + time_stamp+ "_episode%d.h5" %episodes, overwrite=True)
        # self.actor.model.save_weights("trained_model/actormodel_%d.h5" % self.num, overwrite=True)
        # self.critic.model.save_weights("trained_model/criticmodel_%d.h5" % self.num, overwrite=True)

    def save_experiences(self):
        # save experiences of current ddpg
        experiences = open('trained_model/experiences_%d_'%self.num + time_stamp + '.txt', 'w')
        for k in range(self.buff.num_experiences):
            for l in range(29):  # save s_t
                experiences.write(str(self.buff.buffer[k][0][l]) + ';')
            for l in range(3):
                experiences.write(str(self.buff.buffer[k][1][l]) + ';')  # a_t
            experiences.write(str(self.buff.buffer[k][2]) + ';')  # r_t
            for l in range(29):  # s_t1
                experiences.write(str(self.buff.buffer[k][3][l]) + ';')
            experiences.write(str(self.buff.buffer[k][4]))  # done
            experiences.write('\n')
        experiences.close()
