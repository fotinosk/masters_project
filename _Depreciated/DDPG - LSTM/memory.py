import random
import torch

class Buffer:
    """Each sample contains the whole episode"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, history_in, history_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            # ie if not full
            self.buffer.append(None)
        self.buffer[self.position] = (history_in, history_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position+1) % self.capacity)

    def sample(self, batch_size):
        states, actions, last_actions, rewards, next_states, his, cis, hos, cos, dones = [], [], [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)

        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done = sample
            states.append(state)
            actions.append(action)
            last_actions.append(last_action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            his.append(h_in)
            cis.append(c_in)
            hos.append(h_out)
            cos.append(c_out)

        his = torch.cat(his, dim=-2).detach()
        hos = torch.cat(hos, dim=-2).detach()
        cis = torch.cat(cis, dim=-2).detach()
        cos = torch.cat(cos, dim=-2).detach()

        hidden_in = (his, cis)
        hidden_out = (hos, cos)

        return hidden_in, hidden_out, states, actions, last_actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
