# Project 3: Collaboration and Competition

### Introduction

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.



### Solving the Environment
I used DDPG algorithm and this algorithm works great ! 

For Tendency of Graph, I set a higher average score of 0.55.

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the average of these 2 scores.

This yields an average score for each episode (where the average is over all 2 agents).
The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +0.5.


### Observations:
- There is an .ipynb file for jupyter notebook execution.
- The <b>checkpoint.pth</b> has the expected average score already hit.


### Requeriments:
- tensorflow: 1.7.1
- Pillow: 4.2.1
- matplotlib
- numpy: 1.11.0
- pytest: 3.2.2
- docopt
- pyyaml
- protobuf: 3.5.2
- grpcio: 1.11.0
- torch: 0.4.1
- pandas
- scipy
- ipykernel
- jupyter: 5.6.0

### The hyperparameters:
- The file with the hyperparameters configuration is the <b>main.py</b>. 
- If you want you can change the model configuration to into the <b>model.py</b> file.
- The actual configuration of the hyperparameters is: 
  - Learning Rate: 1e-4 (in both DNN)
  - Batch Size: 128
  - Replay Buffer: 1e5
  - Gamma: 0.99
  - Tau: 1e-3
  - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.)

- For the neural models:    
  - Actor  
  
    - Hidden: (input, 1024) - ReLU
    - Hidden: (1024, 512)   - ReLU
    - Hidden: (512, 256)    - ReLU
    - Hidden: (256, 128)    - ReLU
    - Output: (128, 1)      - TanH

  - Critic
 
    - Hidden: (input, 1024)                - ReLU
    - Hidden: (1024 + action_size , 512)   - ReLU
    - Hidden: (512, 256)                   - ReLU
    - Hidden: (256, 128)                   - ReLU
    - Output: (128, action_size)           - linear

### How to train this model?


This code was consisted of 3 part, Tennis.ipynb, model.py.

(Agent implementation is in the Tennis.ipynb because of trial error.)  

If you want to train the code, you need to execute Tennis.ipynb in jupyter notebook.

Pretrained model is also enclosed in zip file, so if you want to check the result of training directly, you can use them.


## Furthermore


you can consider other replay methods, like Prioritized Experience Replay method.

This method is to calculate the importance of the Replay buffer and add weights to the good experence.

you can find references by below links:


A Novel DDPG Method with Prioritized Experience Replay : https://cardwing.github.io/files/DDPG-SMC.pdf

implementation : https://github.com/rlcode/per







