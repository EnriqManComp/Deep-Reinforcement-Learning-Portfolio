from model import ActorCriticNetwork
import torch
import torch.nn.functional as F

class Agent():
    def __init__(self, input_dim, action_dim, learning_rate, gamma=0.99):        
        # Define the actor-critic network
        self.actor_critic_network = ActorCriticNetwork(
            input_obs= [input_dim],
            n_actions= action_dim,
            fc1= 1024,
            fc2= 512,
            learning_rate= learning_rate,
            model_name= "A2C_model",
            save_root_dir= "model/"
        )
        # Set the learning rate and gamma
        self.lr = learning_rate
        self.gamma = gamma
        self.log_prob = None

    def policy(self, state):
        """
        Choose a action based on the state.

        Args:
            state (): The observation of the environment.
        
        Returns:
            action (): The action to take.
        """
        
        # Converting the state to a tensor
        state = torch.tensor(state, dtype=torch.float).to(self.actor_critic_network.device)
        # Get the probabilities of the actions
        probabilities, _ = self.actor_critic_network.forward(state)
        # Convert the probabilities to a probability distribution
        action_probs = torch.distributions.Categorical(probabilities)        
        # Sample the action
        action = action_probs.sample()        
        # Store the log probability
        self.log_prob = action_probs.log_prob(action)        
        # Return the action
        return action.item()
    
    def learn(self, state, reward, next_state, done):

        self.actor_critic_network.optimizer.zero_grad()

        # Convert the states and reward to tensors
        state = torch.tensor(state, dtype=torch.float).to(self.actor_critic_network)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.actor_critic_network)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic_network)

        _, current_value = self.actor_critic_network.forward(state)
        _, next_value = self.actor_critic_network.forward(next_state)

        # Calculate the advantage
        delta = reward + self.gamma * next_value * (1 - int(done)) - current_value        

        # Calculate the actor loss
        actor_loss = -self.log_prob * delta
        # Calculate the critic loss
        critic_loss = delta**2

        # Backward step
        (actor_loss + critic_loss).backward()
        # Optimizer step
        self.actor_critic_network.optimizer.step()        
        


