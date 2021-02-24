from visualizer import Visualizer
from ddpg_deeper import DDPG
from augment import Augment
import gym
import gym_Boeing
import torch
import matplotlib.pyplot as plt

env         = "failure-test-v3"
save_dir    = "./saved_deep_models_failure_modes/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make(env)
augment = Augment(state_size=3, action_size=env.action_space.shape[0])
num_inputs = len(augment)

gamma = 0.99  
tau = 0.001  
hidden_size = (100, 400, 300)  

trained_model  = "failure-train-v3"
checkpoint_dir = save_dir + trained_model

agent = DDPG(gamma, tau, hidden_size, num_inputs, 
             env.action_space, checkpoint_dir=checkpoint_dir)

agent.load_checkpoint()  # TODO: uncomment

v = Visualizer(env, agent)

def see_brain(ds):
    state = torch.Tensor([env.reset(ds = ds)]).to(device)
    augment.reset()

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

        if done or _['len']==100:
            v.save_mat(ds)
            # input('Save then press ENTER to continue...')
            break

pos = env.possibilities
excited_state = 3
num_modes = int(pos/4)

for i in range(num_modes):
    x = i * 4 + excited_state
    v.set_title(f"Mode: {i}, with excited state {excited_state}")
    see_brain(x)
