# Social-Navigation-PyEnvs

> An infrastructure used to simulate crowds of pedestrians and train robots within a social navigation context.

## Description

This repository contains an infrastructure used to simulate crowds of pedestrians moving with different known models (Social Force Model, Headed Social Force Model, ORCA, and many variations) and an OpenAI gym environment useful to train robots to navigate autonomously in indoor crowded environment.

![social-nav-overview-1](.images/social-nav-overview-1.gif)

The simulator is built upon the pygame library in order to provide a functional visualization tool. Here is a list of all the human motion models available:
- Social Force Model (a python re-implementation of the light-sfm library by robotics-upo)
- Social Force Model as defined by Helbing
- Social Force Model with a modification proposed by Guo
- Social Force Model with a modification proposed by Moussaid
- Headed Social Force Model as defined by Farina
- Headed Social Force Model with a modification proposed by Guo
- Headed Social Force Model with a modification proposed by Moussaid
- Modified Headed Social Force Model (new way to compute the torque force driving the pedestrains' heading)
- Modified Headed Social Force Model with a modification proposed by Guo
- Modified Headed Social Force Model with a modification proposed by Moussaid
- Obstacle Reciprocal Collision Avoidance

The simulator also implements a laser sensor and a differential drive for the robot, thus enabling users to develop more low-level applications.

![social-nav-overview-2](.images/social-nav-overview-2.gif)

## Current status



## Table of contents
- [Installation](#installation)
- [Getting started](#getting-started)
- [References](#references)

## Installation
### Prerequisites
### Step-by-step installation guide

## Getting started

## References
- [CrowdNav](https://github.com/ChanganVR/RelationalGraphLearning)
- [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)
- [Light-sfm](https://github.com/robotics-upo/lightsfm)
- Helbing, Dirk, Illés Farkas, and Tamas Vicsek. "Simulating dynamical features of escape panic." Nature 407.6803 (2000): 487-490.
- Guo, Ren-Yong. "Simulation of spatial and temporal separation of pedestrian counter flow through a bottleneck." Physica A: Statistical Mechanics and its Applications 415 (2014): 428-439.
- Moussaïd, Mehdi, et al. "Experimental study of the behavioural mechanisms underlying self-organization in human crowds." Proceedings of the Royal Society B: Biological Sciences 276.1668 (2009): 2755-2762.