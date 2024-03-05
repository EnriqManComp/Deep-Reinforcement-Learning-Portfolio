import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve

if __name__=='__main__':
    # Create the environment
    env = gym.make('MountainCar-v0')        
    
    agent = Agent(
        input_dim= env.observation_space.shape[0],
        action_dim= env.action_space.n,
        learning_rate= 0.0001
    )

    figure_file = 'records/results.png'

    scores = []
    EPISODES = 1000

    for episode in range(EPISODES):
        done = False
        truncated = False
        state = env.reset()
        state = state[0]
        score = 0.0
        while((not done) or (not truncated)):
            action = agent.policy(state)            
            next_state, reward, done, truncated, _ = env.step(action)
            score += reward            
            agent.learn(state, reward, next_state, done)
            state = next_state
    
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print("Episode: ", episode, " Score: %.3f", score, " Average score: %.3f", avg_score)
    
    if avg_score >= 80 and avg_score > 80:
        agent.actor_critic_network.save_checkpoint()
    elif episode % 300 == 0:
        agent.actor_critic_network.save_checkpoint()

    x = [i+1 for i in range(EPISODES)]
    plot_learning_curve(x, scores, figure_file)

    

        
        
    

    