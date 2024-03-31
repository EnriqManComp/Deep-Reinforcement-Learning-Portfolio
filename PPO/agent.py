from model import ActorNetwork
from model import CriticNetwork
import torch as T
from memory import Memory
import numpy as np


class Agent():
    def __init__(self, input_dims, action_dim, learning_rate=0.0003, gae_lambda=0.95, gamma=0.99, policy_clip=0.2, batch_size=64,
                 n_epochs=10):        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions=action_dim,
                                  input_dims= input_dims,
                                   learning_rate= learning_rate)
        self.critic = CriticNetwork(input_dims=input_dims,
                                    learning_rate= learning_rate)
        self.memory = Memory(batch_size)
        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... save models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... load models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def policy(self, state):
        """
        Choose a action based on the state.

        Args:
            state (): The observation of the environment.
        
        Returns:
            action (): The action to take.
        """
        
        # Converting the state to a tensor
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        # Get the probabilities of the actions
        prob_dist = self.actor(state)
        value = self.critic(state)
        # Sample the action
        action = prob_dist.sample()        
        
        probs = T.squeeze(prob_dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        
        return action, probs, value
    
    def learn(self):

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
            reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma*values[k+1]* \
                    (1-int(dones_arr[k])) - values[k])

                    discount *= self.gamma*self.gae_lambda
                
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                distributions = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = distributions.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                                                 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()








              



