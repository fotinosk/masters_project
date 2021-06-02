import numpy as np
import torch
from scipy.signal import resample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Augment:
    def __init__(self, state_size, action_size, memory_size=40, output_size=11):
        """
        Takes as input the current state and outputs an augmented state
        Augmented state = [x_k u_k-1 x_k-1 ...] (MUST BE FLAT!)
        The state is then fed in the action, giving the next action u_k,
        which then updates the augmented state
        It is important to average down from the memory size to the output size

        Each output tensor will contain output_size state vectors and output_size-1 action vectors

        Args:
            state_size (int): size of the state
            action_size (int): size of the action 
            memory_size (int, optional): Memory size, ie how many past states and actions to store. Defaults to 40.
            output_size (int, optional): How many past states and actions to output, this is done by averaging. Defaults to 11.
        """        
        self.memory_size = memory_size
        self.output_size = output_size
        self.action_size = action_size
        self.state_size  = state_size

        self.output_len = output_size * state_size + (output_size-1) * action_size

        # Once filled it will be a list of lists of [x,u] <- both tensors
        self.memory = [[np.zeros(state_size), np.zeros(action_size)]]*memory_size

        self.last_state = None
        self.last_augmented = None
        self.last_action = None

    def __call__(self, state):
        """Receives a state and outputs an augmented state. Calls the average function and stores the state in the last state

        Args:
            state (tensor)
        """  
        self.last_state = state      
        state_array, action_array = self.average()

        state_array = np.append(state.cpu().numpy(), state_array).astype(np.float32)
        action_array = np.append([], action_array).astype(np.float32)

        augmented_state = np.append(state_array, action_array)

        self.last_augmented = augmented_state

        return torch.tensor(augmented_state).to(device)

    def update(self, action):
        """Updates memory, adds the [last_state, action] pair to memory and removes the oldest one
        Converts them to numpy arrays before adding them, for easier subsampling

        Args:
            action (tensor)
        """  
        self.memory.pop(-1)      
        self.memory.insert(0, [self.last_state.cpu().numpy(), action.cpu().numpy()])

        self.last_action = action

    def average(self):
        """Avarages memory, to downsample the output to the output size
        """

        x = list(zip(*self.memory))
        states  = list(x[0])
        actions = list(x[1])
        
        downsampled_states  = resample(states , self.output_size-1)
        downsampled_actions = resample(actions, self.output_size-1)

        return downsampled_states, downsampled_actions

    def mock_augment(self, next_state, aug_state, action):
        """Used in the replay buffer when updating the parameters
        
        DO NOT USE ANY OF THE EXISTING VALUES, SINCE THIS IS ASYNC!

        Args:
            next_state (tensor)
            aug_state (tensor)
            action (tensor)
        """        

        # print(aug_state, action, next_state)
        next_state = torch.tensor(next_state).to(device)

        states = torch.hstack((next_state, aug_state[:self.state_size*(self.output_size-1)]))
        actions = torch.hstack((action, aug_state[self.state_size*self.output_size:-self.action_size]))

        return torch.hstack((states, actions)).cpu().numpy()

    def reset(self):
        self.memory = [[np.zeros(self.state_size), np.zeros(self.action_size)]]*self.memory_size
        self.last_state = None
        self.last_augmented = None
        self.last_action = None
        

    def __len__(self):
        return self.output_len
