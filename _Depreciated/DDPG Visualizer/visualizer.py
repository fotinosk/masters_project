"""A class used to visualize the activations of a DDPG network as the episode progresses"""

class Visualizer:
    """
    Visualizes one episode at a time
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.actor = agent.actor
        self.actor_target = agent.actor_target
        self.critic = agent.critic
        self.critic_target = agent.critic_target