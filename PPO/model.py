import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, learning_rate, \
                 fc1_dims=256, fc2_dims=256, save_root_dir='model/'):
        super(ActorNetwork, self).__init__()
        
        # Set the model name and save root directory        
        self.save_dir = os.path.join(save_root_dir, "actor_network_ppo")

        # Common layers
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )    

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # GPU configuration
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)

    def forward(self, state):       

        distribution = self.actor(state)
        final_distribution = Categorical(distribution)
        
        return final_distribution

    def save_checkpoint(self):
        """
        Save the model checkpoint.
        """        
        T.save(self.state_dict(), self.save_dir)    
        print(f"Model saved!!!")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_dir))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate, fc1_dims=256, fc2_dims=256, chkpt_dir='model/'):
        super(CriticNetwork, self).__init__()

        self.save_dir = os.path.join(chkpt_dir, "critic_network_ppo")
        
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):

        value = self.critic(state)

        return value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_dir)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_dir))




