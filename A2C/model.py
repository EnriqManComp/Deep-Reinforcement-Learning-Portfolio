import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_obs, n_actions, fc1, fc2, learning_rate, model_name, save_root_dir):
        super(ActorCriticNetwork, self).__init__()
        
        # Set the model name and save root directory
        self.model_name = model_name
        self.save_dir = os.path.join(save_root_dir, model_name)

        # Common layers
        self.dense1 = nn.Linear(*input_obs, fc1)
        self.dense2 = nn.Linear(fc1, fc2)

        # Actor layer
        self.actor = nn.Linear(fc2, n_actions)        

        # Critic layer
        self.critic = nn.Linear(fc2, 1)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # GPU configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Generate a step forward in the networks.

        Args:
            state (torch.Tensor): The observation of the environment.

        Returns:
            tuple: A tuple containing the value of the state (V) and the numerical results of the actor layer (actor_output).
        """
        
        # Common layers
        X = F.relu(self.dense1(state))
        X = F.relu(self.dense2(X))

        # Actor layer
        # Action probabilities        
        policy_dist = F.softmax(self.actor(X), dim=0)               

        # Critic layer
        v = self.critic(X)

        return policy_dist, v

    def save_checkpoint(self):
        """
        Save the model checkpoint.
        """        
        torch.save(self.state_dict(), self.save_dir)    
        print(f"Model saved!!!")