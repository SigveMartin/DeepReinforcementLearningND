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

#### Deep Q-Network
In this project the agent learns from 37 dimmensions of the environment (as described above), and uses a deep neural network (as found in [model.py](model.py)) to map to the four avaiable actions.

The neural network uses three fully connected [linear layers](https://pytorch.org/docs/stable/nn.html#linear) with [rectified linear unit activation](https://pytorch.org/docs/stable/nn.html#torch.nn.functional.relu) between the fully connected hidden layers and returns an vector of action values from the third fully connected linear layer.

#### Agent
The agent is defined in [dqn_agent.py](dqn_agent.py), which uses the deep q-network.

#### Experience replay
Our agent uses experience [replay buffer](https://github.com/SigveMartin/drlnd/blob/22c0a477933d6c0e8b72c8ab45f74173025badd4/project_1_Navigation/dqn_agent.py#L120) as illustrated in the image below.

![exerience replay](images/replaybuffer.png)

The agent interacts with the environment and collects experience tuples. These are stored in the replay buffer ([the agents memory](https://github.com/SigveMartin/drlnd/blob/22c0a477933d6c0e8b72c8ab45f74173025badd4/project_1_Navigation/dqn_agent.py#L41)) and later randomly sampled in order to help break the correlations between consecutive experiences and help stabilize the learning algoritm.

In our network we store experience for every step, and then learns (update the network) every fourth step.

#### Double DQN
In order to prevent overetimation of q-values this agent is equipped with two function approximations, or a double set of [q-networks (local and target)](https://github.com/SigveMartin/drlnd/blob/22c0a477933d6c0e8b72c8ab45f74173025badd4/project_1_Navigation/dqn_agent.py#L36).

![double DQN](images/doubleq.png)

In our implemention we get the maximum predicted Q values from the target model and the expected q-values from the local model and evaluate them. The [target model is updated](https://github.com/SigveMartin/drlnd/blob/22c0a477933d6c0e8b72c8ab45f74173025badd4/project_1_Navigation/dqn_agent.py#L106) by copying the weights from the local model into the target model, when learning from the experiences.

The local model is used to produce actions for a state when the agent acts. However, these two function approximators must agree on the best actions through the learning step on the Q-values.

In the long run this prevents the algoritm from propogating incidential high rewards that might be obtained by chance and don't reflect long-turn returns (ref. lesson 2.9 Udacity).

#### Hyperparameters for DQN
The hyperparameters for the agent is found and explained in the [dqn_agent.py](dqn_agent.py).
>`
* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 64         # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR = 5e-4               # learning rate
* UPDATE_EVERY = 4        # how often to update the network`



## Plot of Rewards

The project was trained on 1000 episodes. As the plot of rewards show below, it could have

![Plot of rewards](images/projec_1_results.png)

As the plot shows, 600 episode had been sufficient as the score reached over 13 on average and that it didn't increase significantly after that.

## Ideas for future work
