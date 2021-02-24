"""A class used to visualize the activations of a DDPG network as the episode progresses"""

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.widgets import Button
import torch.nn as nn
import pandas as pd
import numpy as np

class Visualizer:
    """
    Visualizes one episode at a time
    """

    def __init__(self, env, agent):
        # Target actor and critic are omitted since they are not used in testing
        self.env = env
        self.agent = agent
        self.actor = agent.actor
        self.critic = agent.critic
        self.activations_actor = {}
        self.activations_critic = {}
        self.weights_actor = []
        self.weights_critic = []
        self.title = 'Placeholder'

        for i, m in enumerate(self.actor.modules()):
            if i != 0 and isinstance(m, nn.Linear):
                weight = m.weight.data.cpu().numpy()
                self.weights_actor.append(weight)
                m.register_forward_hook(self.get_activation_actor('Actor layer ' + str(i)))
        
        for i, m in enumerate(self.critic.modules()):
            if i != 0 and isinstance(m, nn.Linear):
                weight = m.weight.data.cpu().numpy()
                self.weights_critic.append(weight)
                m.register_forward_hook(self.get_activation_critic('Critic layer ' + str(i)))

        self.fig, self.ax = plt.subplots(nrows=2, ncols=len(self.weights_actor), figsize=(8, 5))
        self.fig.suptitle(self.title)
        self.fig.tight_layout()
        plt.subplots_adjust(top=0.85)

        for i in range(len(self.activations_actor)):
            self.ax[0,i].set_title(f'Actor: Layer {i}')
            self.ax[1,i].set_title(f'Critic: Layer {i}')


        # self.fig, self.ax = plt.subplots(nrows=1, ncols=len(self.weights_actor) ,figsize=(8, 5))
        # self.fig.suptitle('Actor', fontsize=20)

        # self.fig2, self.ax2 = plt.subplots(nrows=1, ncols=len(self.weights_critic) ,figsize=(8, 5))
        # self.fig2.suptitle('Critic', fontsize=20)


    def get_activation_actor(self, name):
        def hook(model, input, output):
            self.activations_actor[name] = output.detach()
        return hook

    def get_activation_critic(self, name):
        def hook(model, input, output):
            self.activations_critic[name] = output.detach()
        return hook

    def prints(self,action):

        ars = []

        for k, item in enumerate(self.activations_actor.items()):
            # self.ax[k].cla()
            # self.ax[k].axis('off')
            # ar = 0.1 * 100/(len(item[1]))
            # cax = self.ax[k].matshow(item[1].cpu().reshape((-1,1)), aspect=ar, cmap = cm.bwr)
            # self.ax[k].set_title(item[0])

            self.ax[0, k].cla()
            self.ax[0, k].axis('off')
            ar = 0.01 * 100/(len(item[1]))
            ars.append(ar)
            cax = self.ax[0, k].matshow(item[1].cpu().reshape((-1,1)), aspect=ar, cmap = cm.bwr)
            self.ax[0, k].set_title(item[0])

        for k, item in enumerate(self.activations_critic.items()):
            # self.ax2[k].cla()
            # self.ax2[k].axis('off')
            # ar = 0.1 * 100/(len(item[1][0]))
            # cax2 = self.ax2[k].matshow(item[1][0].cpu().reshape((-1,1)), aspect=ar, cmap = cm.bwr)
            # self.ax2[k].set_title(item[0])

            self.ax[1, k].cla()
            self.ax[1, k].axis('off')
            # ar2 = 0.1 * 100/(len(item[1]))
            ar2 = ars[k]
            cax2 = self.ax[1, k].matshow(item[1].cpu().reshape((-1,1)), aspect=ar2, cmap = cm.bwr)
            self.ax[1, k].set_title(item[0])

        # plt.matshow(self.weights_actor.reshape((1,-1)))
        # textstr = "Action: " + str(action[-3:])
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # self.ax[0].text(0, -0.1, textstr, transform=self.ax[0].transAxes, fontsize=14,
        #     verticalalignment='bottom', bbox=props)
        # self.ax2[0].text(0, -0.1, textstr, transform=self.ax2[0].transAxes, fontsize=14,
        #     verticalalignment='bottom', bbox=props)

        # self.ax[0,0].text(0, -0.1, textstr, transform=self.ax[0,0].transAxes, fontsize=14,
        #     verticalalignment='bottom', bbox=props)
        # self.ax[1,0].text(0, -0.1, textstr, transform=self.ax[1,0].transAxes, fontsize=14,
        #     verticalalignment='bottom', bbox=props)
        ars = []
        plt.draw()
        plt.pause(0.0001)

    def set_title(self, title):
        self.title = title
        self.fig.suptitle(self.title)

    def save_mat(self, ds):
        for item in self.activations_actor.items():
            mat = item[1].cpu().numpy()
            mat = np.asarray(mat)
            # np.savetxt(ds+ item[0] + '.csv', mat, delimiter=',')
            np.savetxt(f"Visualizer/activations/{ds} {item[0]}.csv", mat, delimiter=',')
        for item in self.activations_critic.items():
            mat = item[1].cpu().numpy()
            mat = np.asarray(mat)
            # np.savetxt(ds + item[0] + '.csv', mat, delimiter=',')
            np.savetxt(f"Visualizer/activations/{ds} {item[0]}.csv", mat, delimiter=',')