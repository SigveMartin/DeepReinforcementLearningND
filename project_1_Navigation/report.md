# Report Project 1: Navigation

For this project, an agent is trained to navigate (and collect bananas!) in a large, square world.

The project code is contained in three seperate files.

* `navigation_notebook.ipynb`: Notebook you test and run the code.
* `dqn_agent.py`: python class where the agent is defined.
* `model.py`: class where the Q-network is defined, which is used by dqn-agent.py.

The following headings are target to answer [these project rubric points for the report.](https://review.udacity.com/#!/rubrics/1889/view)

## Project Setup
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Learning Algoritm
This project uses reinforcement learning, where an agent learns from an environment, to figure out the best actions to take. It does so by making actions and perceive observations from the environment.

![reinforcement learning](images/dqn-architecture.png).

Reinforcement learning is often described as an agent trying to interact with an previously unknown environment, trying to maximize the overall or total reward. This learning algoritm is using deep neural network to approxmate the function that map the observations to the right actions to take in order to maximize this reward.

![Deep rl](images/deep reinforcement learning.png).

In this project the agent learns from 37 dimmensions of the environment (as described above), and uses a deep neural network (as found in [model.py](/model.py)) to map to the four avaiable actions.

 ![Deep nn](images/deep reinforcement learning.png).

The neural network uses three [linear layers](https://pytorch.org/docs/stable/nn.html#linear) with [rectified linear unit activation](https://pytorch.org/docs/stable/nn.html#torch.nn.functional.relu) between layers and returns an action vector with weights corresponding to the action space the agent have in its environment.




## Plot of Rewards

The project was trained on 1000 episodes. As the plot of rewards show below, it could have

![Plot of rewards](images/projec_1_results.png)

As the plot shows, 600 episode had been sufficient as the score reached over 13 on average and that it didn't increase significantly after that.

## Ideas for future work
