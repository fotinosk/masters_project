from the_golfer import golfer, augment, agentA, agentB, device
import gym
import gym_Boeing
import torch

env = gym.make('failure-train-v2')  # TODO: make new environment

save_dir = r"./saved_deep_models_golfer/"
checkpoint_dir = save_dir + env
filename = 'runs/run_' + datetime.datetime.now().strftime("%m%d%H%M")
writer = SummaryWriter(filename)

while timestep <= 1e7:
    epoch_return = 0

    state = torch.tensor([env.reset()]).to(device)
    augment.reset()

    while True:
        state = augment(state[0]).to(device)
        action = golfer.forward(state).to(device)

        next_state, reward, done, _ = env.step(action.cpu().numpy())
        augment.update(action)

        writer.add_scalar('Reward', reward, timestep)

        timestep += 1
        epoch_return += reward

        state = next_state # this might not work