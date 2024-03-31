import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve

if __name__=='__main__':
    # Create the environment
    env = gym.make('CartPole-v0')        
    
    N = 20
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

    n_games = 300

    figure_file = 'results/results.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    for i in range(n_games):        
        
        done = False      
        truncated = False  
        
        state = env.reset()        
        state = state[0]
        
        score = 0.0
               
        while (not done):
                                         
            action, prob, val = agent.policy(state)           
            next_state, reward, done, truncated, _ = env.step(action)
            n_steps +=1
            score += reward   
            
            agent.remember(state, action, prob, val, reward, done)              
            
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1 
            state = next_state            
    
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print("Episode: ", i, " Score: %.3f" % score, " Average score: %.3f" % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

    

        
        
    

    