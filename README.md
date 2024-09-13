# Social-Navigation-PyEnvs

> An infrastructure used to train robots within a social navigation context with a wide range of human motion models to simulate crowds of pedestrians.

## Description

This repository contains an infrastructure developed starting from CrowdNav <a href="#crowdnav">[1]</a> and Python-RVO2 <a href="#pythonrvo2">[2]</a> used to train and test learning-based algorithms for Social Navigation.

In order to simulate crowds of pedestrians the following models are implemented:
- Social Force Model (SFM) <a href="#sfm">[3]</a> and its variations <a href="#sfm_moussaid">[4]</a>, <a href="#sfm_guo">[6]</a>.
- Optimal Reciprocal Collision Avoidance (ORCA) <a href="#orca">[5]</a>.
- Headed Social Force Model (HSFM) <a href="#hsfm">[7]</a>.

The CrowdNav module <a href="#crowdnav">[1]</a> includes the following reinforcement learning alogrithms for social robot navigation:
- Collision Avoidance with Deep RL (CADRL) <a href="#cadrl">[8]</a>.
- Long-short term memory RL (LSTM-RL) <a href="#lstmrl">[9]</a>.
- Social Attentive RL (SARL)<a href="#sarl">[10]</a>.

The simulator is built upon [Pygame](https://www.pygame.org/) in order to provide a functional visualization tool and [OpenAI Gym](https://gymnasium.farama.org/), which defines the API standard for RL environments.

![social-nav-overview-1](.images/social-nav-overview-1.gif) 

The simulator also implements a laser sensor and a differential drive for the robot, thus enabling users to develop more low-level applications.

![social-nav-overview-2](.images/social-nav-overview-2.gif)

## References
<ol>
    <li id="crowdnav"><a href="https://github.com/ChanganVR/RelationalGraphLearning">CrowdNav</a></li>
    <li id="pythonrvo2"><a href="https://github.com/sybrenstuvel/Python-RVO2">Python-RVO2</a></li>
    <li id="sfm">Helbing, Dirk, Illés Farkas, and Tamas Vicsek. "Simulating dynamical features of escape panic." Nature 407.6803 (2000): 487-490.</li>
    <li id="sfm_moussaid">Moussaïd, Mehdi, et al. "Experimental study of the behavioural mechanisms underlying self-organization in human crowds." Proceedings of the Royal Society B: Biological Sciences 276.1668 (2009): 2755-2762.</li>
    <li id="orca">Van Den Berg, Jur, et al. "Reciprocal collision avoidance with acceleration-velocity obstacles." 2011 IEEE International Conference on Robotics and Automation. IEEE, 2011.</li>
    <li id="sfm_guo">Guo, Ren-Yong. "Simulation of spatial and temporal separation of pedestrian counter flow through a bottleneck." Physica A: Statistical Mechanics and its Applications 415 (2014): 428-439.</a></li>
    <li id="hsfm">Farina, Francesco, et al. "Walking ahead: The headed social force model." PloS one 12.1 (2017): e0169734.</li>
    <li id="cadrl">Chen, Yu Fan, et al. "Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning." 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 2017.</li>
    <li id="lstmrl">Everett, Michael, Yu Fan Chen, and P. Jonathan. "How. Motion planning among dynamic, decision-making agents with deep reinforcement learning. In 2018 IEEE." RSJ International Conference on Intelligent Robots and Systems (IROS).</li>
    <li id="sarl">Chen, Changan, et al. "Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning." 2019 international conference on robotics and automation (ICRA). IEEE, 2019.</li>
</ol>