import gym
from agent import Agent

if __name__='__main__':
    # Create the environment
    env = gym.make('BipedalWalker-v3')
    print(env.observation_space.shape[0])
    '''
    agent = Agent(
        input_dim= env.observation_space.shape[0],
        action_dim= env.action_space.n,
        learning_rate= 0.0001
    )
    '''