from visualizer import Visualizer
from ddpg_deeper import DDPG
from augment import Augment
import gym
import gym_Boeing
import torch

env         = "failure-test-v1"
save_dir    = "./saved_deep_models_failure_modes/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make(env)
augment = Augment(state_size=3, action_size=env.action_space.shape[0])
num_inputs = len(augment)

gamma = 0.99  
tau = 0.001  
hidden_size = (100, 400, 300)  

trained_model  = "failure-train-v1"
checkpoint_dir = save_dir + trained_model

agent = DDPG(gamma, tau, hidden_size, num_inputs, 
             env.action_space, checkpoint_dir=checkpoint_dir)

agent.load_checkpoint()

state = torch.Tensor([env.reset()]).to(device)
v = Visualizer(env, agent)

while True:
    state = augment(state[0])
    action = agent.calc_action(state, action_noise=None).to(device)

    if state.dim() == 1:
        state = state.unsqueeze(0).to(device)
    if action.dim() == 1:
        action = action.unsqueeze(0).to(device)

    q_value = agent.critic(state, action)
    next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

    augment.update(action[0])

    state = torch.Tensor([next_state]).to(device)

    v.prints(action.cpu().numpy()[0])

    if done:
        break

    