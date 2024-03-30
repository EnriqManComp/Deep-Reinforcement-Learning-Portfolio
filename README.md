# Deep-Reinforcement-Learning-Algorithm

<details>
<summary> PPO </summary>
  
**<h3>Motivation:</h3>**

1) Actor Critic methods are sensitive to perturbations.
2) Limits update to policy network.
3) Base the update on the ratio of new policy to old.
4) Have to account for goodness of state (advantage).
5) Clip loss function and take lower bound with min.
6) Keeps track of a fixed length trajectory of memories.
7) Uses multiple network updates per data sample.
   * Minibatch stochastic gradient ascent.

**<h3>Implementation details</h3>**
1) Critic evaluates states (not s,a pairs)
2) Actor decides what to do based on current state.


</details>
