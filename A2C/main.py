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

    figure_file = 'results.png'

    scores = []
    EPISODES = 2000

    save_net_count = 1

    load = True
    if load:
        agent.load_model()       
        #save_net_count =

    for episode in range(1, EPISODES):        
        done = False
        truncated = False
        state = env.reset()
        state = state[0]
        score = 0.0        
        while((not done) and (not truncated)):            
            action = agent.policy(state)           
            next_state, reward, done, truncated, _ = env.step(action)
            score += reward            
            agent.learn(state, reward, next_state, done)
            state = next_state            
    
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print("Episode: ", episode, " Score: %.3f" % score, " Average score: %.3f" % avg_score)
    
        if score >= -100 and avg_score > -100:
            agent.actor_critic_network.save_checkpoint()
            with open("save_network.txt", 'a') as file:
                file.write("Save: {0}, Episode: {1}/{2}, Score mean: {3}\n".format(save_net_count, episode, EPISODES, avg_score))
        elif episode % 300 == 0:
            agent.actor_critic_network.save_checkpoint()
            with open("save_network.txt", 'a') as file:
                file.write("Save: {0}, Episode: {1}/{2}, Score mean: {3}\n".format(save_net_count, episode, EPISODES, avg_score))
        if episode % 100 == 0:
            with open("record.txt", 'a') as file:
                file.write("Scores: {0}\n".format(scores))
    x = [i+1 for i in range(EPISODES)]
    plot_learning_curve(x, scores, figure_file)

    

        
        
    

    