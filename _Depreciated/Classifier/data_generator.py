import numpy as np
import torch
import gym
import gym_Boeing
from graph import imagify
from augment import Augment
from ddpg_deeper import DDPG

classifier_env = gym.make('failure-train-v2')
agent_env      = gym.make('failure-train-v1')
save_dir    = "./saved_deep_models_failure_modes/"
graph = imagify(classifier_env.action_space.shape[0], 
                classifier_env.observation_space.shape[0])

# Initialize agent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gamma = 0.99  
tau = 0.001  
hidden_size = (100, 400, 300)  

augment = Augment(state_size=3, action_size=agent_env.action_space.shape[0])
num_inputs = len(augment)

trained_model = 'failure-train-v1'
checkpoint_dir = save_dir + trained_model

agent = DDPG(gamma,tau,hidden_size,num_inputs, agent_env.action_space, checkpoint_dir)
# agent.load_checkpoint()
agent.set_eval()

class graph_generator:

    # agent is asked to control the classifier environment to generate both good and bad graphs
    def __init__(self, agent, env, graph):
        self.agent = agent
        self.graph = graph
        self.env   = env
        self.possibilities = self.env.possibilities
        self.num_modes = int(self.possibilities / 4)

    def get_num_labels(self):
        return self.num_modes

    def generate_graph(self):
        self.graph.reset()
        state = np.random.randint(0, self.possibilities)
        label = state // 4
        exc_state = state % 4

        stop_point = np.random.randint(50, 1000)

        state = torch.Tensor([self.env.reset(ds=state)]).to(device)
        for step in range(stop_point):
            astate = augment(state[0])
            action = agent.calc_action(astate).to(device)

            if astate.dim() == 1:
                astate = astate.unsqueeze(0).to(device)
            if action.dim() == 1:
                action = action.unsqueeze(0).to(device)
            
            self.graph.update(action.cpu(), state.cpu())
            next_state, r, d, _ = self.env.step(action.cpu().numpy()[0])
            augment.update(action[0])

            state = torch.Tensor([next_state]).to(device)
        
        return (label, self.graph())


Generator = graph_generator(agent, classifier_env, graph)

# for i in range(100):
#     print(gen.generate_graph())