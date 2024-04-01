import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.distributions.categorical import Categorical

######## NETWORK ARCHITECTURE
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, learning_rate, save_root_dir='./model/'):
        super(ActorNetwork, self).__init__()
        
        # Set the model name and save root directory        
        self.save_dir = os.path.join(save_root_dir, "actor_network_ppo")
        self.n_actions = n_actions
        # Common layers
        self.input_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()            
        )

        self.input_2 = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()            
        )

        self.common_layers = nn.Sequential(
            nn.Linear(3200, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
            nn.Softmax(dim=-1)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # GPU configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)

    def forward(self, state):    

        X1 = self.input_1(state[0])
        X1 = X1.view(X1.size(0), -1)

        X2 = self.input_2(state[1])
        X2 = X2.view(X2.size(0), -1)

        X = torch.cat([X1, X2], dim=1)

        X = self.common_layers(X)   

        distribution = Categorical(X)
        
        return distribution

    def save_checkpoint(self):
        """
        Save the model checkpoint.
        """        
        T.save(self.state_dict(), self.save_dir)    
        print(f"Model saved!!!")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_dir))

class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, chkpt_dir='./model/'):
        super(CriticNetwork, self).__init__()

        self.save_dir = os.path.join(chkpt_dir, "critic_network_ppo")

        self.input_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()            
        )

        self.input_2 = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()            
        )

        self.common_layers = nn.Sequential(
            nn.Linear(3200, 256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        
        X1 = self.input_1(state[0])
        X1 = X1.view(X1.size(0), -1)
        
        X2 = self.input_2(state[1])
        X2 = X2.view(X2.size(0), -1)
        
        X = torch.cat([X1, X2], dim=1)

        value = self.common_layers(X)
        
        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.save_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.save_dir))

    


class DRL_algorithm:

    def __init__(self, memory, replay_exp_initial_condition):

        self.training_finished = False
        self.update_network_counter = 1
        
        
        self.memory = memory
               
        
        # Batch size
        self.batch_size = 32
        # Number of action of the agent
        self.action_dim = 9
        
        self.discount = 0.99        
        
        self.p_action = 0
        self.same_action_counter = 0

        ####### MODELS        
        
        self.actor = ActorNetwork(self.action_dim, learning_rate=0.0003)          
        self.critic = CriticNetwork(learning_rate=0.0003)                       

        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 10
        self.gae_lambda = 0.95                        

    def policy(self, state, lidar_state):
                                            
        # Preprocessing state images                        
        img_state = Image.fromarray(state).convert('L') 
        img_state = img_state.rotate(-90)   
        img_state = img_state.transpose(Image.FLIP_LEFT_RIGHT)          
        img_state = img_state.resize((84, 84))                
        img_state = np.array(img_state) / 255.0            
        

        # Expand dimensions             
        img_state_tensor = torch.tensor(img_state).to(self.actor.device)
        img_state_tensor = img_state_tensor.unsqueeze(0)
        img_state_tensor = img_state_tensor.unsqueeze(0)
        # Lidar state
        lidar_state = np.array(lidar_state) / 200.0
        
        lidar_state_tensor = torch.tensor(lidar_state).to(self.actor.device)    
        lidar_state_tensor = lidar_state_tensor.unsqueeze(0)
        # Prediction            

        prob_dist = self.actor([img_state_tensor.float(), lidar_state_tensor.float()])
        
        value = self.critic([img_state_tensor.float(), lidar_state_tensor.float()])
        
        action = prob_dist.sample()

        probs = torch.squeeze(prob_dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()           
        return action, probs, value
    
    def train(self):
                  
        # Sampling minibatch                
        for _ in range(self.n_epochs):
            img_state_arr, lidar_state_arr, action_arr, old_probs_arr, vals_arr, \
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
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                img_states = torch.tensor(img_state_arr[batch], dtype=torch.float).to(self.actor.device)
                lidar_states = torch.tensor(lidar_state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                distributions = self.actor([img_states, lidar_states])
                critic_value = self.critic([img_states, lidar_states])

                critic_value = torch.squeeze(critic_value)

                new_probs = distributions.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                                                 1+self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

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

        print("Training... ")        
        
        self.training_finished = True
        print("Training Finished")

    def save_model(self, f):        
        self.actor.save_checkpoint(f)
        self.critic.save_checkpoint(f)
        print("Models saved")
    
    def load_models(self, f):  
        self.actor.load_checkpoint(f)
        self.critic.load_checkpoint(f)      
        print("Models loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
