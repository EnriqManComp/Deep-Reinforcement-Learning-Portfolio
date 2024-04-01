import numpy as np

class Memory:
    def __init__(self, batch_size):
        self.img_states = []
        self.lidar_states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.img_states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.img_states),\
                np.array(self.lidar_states), \
                np.array(self.actions),\
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.rewards), \
                np.array(self.dones),\
                batches
    
    def store_memory(self, img_state, lidar_state, action, probs, vals, reward, done):
        self.img_states.append(img_state)
        self.lidar_states.append(lidar_state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.img_states = []
        self.lidar_states = []
        self.probs = []
        self.actions = []
        self.vals = []
        self.rewards = []
        self.dones = []
    