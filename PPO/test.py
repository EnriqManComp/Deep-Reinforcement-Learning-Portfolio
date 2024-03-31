import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve

if __name__=='__main__':
    # Create the environment
    env = gym.make('CartPole-v0')        
    
    
    batch_size = 5
    n_epochs = 4
    learning_rate = 0.0003

    agent = Agent(
        input_dims= env.observation_space.shape[0],
        action_dim= env.action_space.n,
        learning_rate= learning_rate,
        batch_size= batch_size,
        n_epochs= n_epochs
    )

    agent.actor.load_checkpoint()
    agent.critic.load_checkpoint()

    n_games = 300

    figure_file = 'results/test_results.png'

    best_score = env.reward_range[0]
    score_history = []

    avg_score = 0    
    
    for i in range(n_games):        
        
        done = False              
        
        state = env.reset()        
        state = state[0]
        
        score = 0.0
               
        while (not done):
                                         
            action, _, _ = agent.policy(state)           
            next_state, reward, done, _, _ = env.step(action)
            
            score += reward                         
            
            state = next_state            
    
        score_history.append(score)        

        print("Episode: ", i, " Score: %.3f" % score)                    
        
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

    

        
        
    

    