## Ballbot-with-DRL

This repo contains the source code for paper:

- Learning  Ball-balancing  Robot  Through  Deep  Reinforcement  Learning. ICCCR 2021

### Requiremetns

- tensorflow 1.14.0
- gym 0.17.0
- pybullet 2.8.4
- [stable-baselines](https://github.com/hill-a/stable-baselines)

### File Description

The source code can be divided into two types:

- Files used to train the DDPG model by contacting with vrep via GUI mode
  - [training_DDPG.py](training_DDPG.py): main script to conduct the DDPG training
  - [training_env.py](training_env.py): the environment that connects the DDPG agent and the vrep simulation (GUI mode)
  - [simple_enjoy.py](simple_enjoy.py): load a saved DDPG model and run it in vrep simulation environment for (GUI mode) visualization
  - [ballbot_for_train.ttt](ballbot_for_train.ttt): ttt scence file for vrep to open during training time
- Files used to conduct auto test for saved DDPG models by contacting with vrep via HeadLess mode
  - [autotest_DDPG.py](autotest_DDPG.py): load a saved DDPG model and test it with different initial states automatically in vrep simulation (HeadLess mode)
  - [autotest_traditional.py](autotest_traditional.py): test the traditional controller with different initial states automatically in vrep simulation (HeadLess mode)
  - [autotest_env.py](autotest_env.py): the environment that connects the DDPG agent and the vrep simulation (HeadLess mode)
  - [ballbot_for_autotest.ttt](ballbot_for_autotest.ttt): ttt scence file for vrep to open during test time

### How to achieve automatical test for vrep

We can bridge python script and vrep scence by a shared txt file. At each step, we first use python file IO to wrtie the preferred initial state into the shared txt file and then send ```reset``` signal to the vrep process. After being reset, the vrep process reads initialization information in the shared txt file and starts simulation. In this way, we can control the initial state of simulation while the content of ttt scence file remains unchanged.

### Quick start

Clone the stable-baseline repo and put python files to the main directory of stable-baseline. To train a DDPG model, first launch a vrep in GUI mode, and then run command:

```c
python training_DDPG.py
```

To automatically test a DDPG based controller, first make sure the saved model file is in a correct directory, and then run command:

```c
python autotest_DDPG.py
```

### Demo videos

We provide four demo videos to show the performance of traditional/DDPG controller in vrep simulated environment, starting at different initial positions. Videos are in ```./demo_videos``` directory. 