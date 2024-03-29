# Reinforcement Learning on Autonomous Race Car

Reinforcement Learning approach to "Formula Student Technion Driverless" project, simulated in Unreal Engine 4 with AirSim plugin, using Soft Actor Critic (SAC) algorithm and Variational Auto Encoder (VAE).

[Youtube Video](https://www.youtube.com/watch?v=2mk49rsoXf8)

[Full Project Report](https://gip.cs.technion.ac.il/projects/uploads/180_preport_7.pdf)


![gifToGit](https://user-images.githubusercontent.com/38940464/109778141-26a99880-7c0d-11eb-801d-807bf91cded4.gif)

## Table of Contents

- [Prerequisites](#Prerequisites)
- [How To Build](#How-To-Build)
- [Run in Test Mode](#Run-in-Test-Mode)
- [Run in Train Mode](#Run-in-Train-Mode)
- [Run in Train Mode + VAE Training](#Run-in-Train-Mode-+-VAE-Training)
- [Overview](#Overview)
- [Citation](#Citation)
- [Credits](#Credits)


## Prerequisites 
* Operating System: Ubuntu 18.04 or Windows 10
* Software: Unreal Engine 4.24.3
* GPU: Nvidia GTX 1080 or higher (recommended)
  
## How To Build
1. Set up Unreal Engine 4, AirSim and the FSTD Environment by following this link [Ubuntu](https://github.com/eliork/Reinforcement-Learning-on-Autonomous-Race-Car/blob/main/build_FSTDriverless_ubuntu.md) , [Windows](https://github.com/FSTDriverless/AirSim/blob/master/docs/build_FSTDriverless_windows.md)
  2. If you are using Ubuntu, you can skip this step. If you are using Windows: Download updated RaceCourse folder from this [link](https://drive.google.com/file/d/1lpBHRYYw7GRICgLaSfMQcXlbP2A98b9L/view?usp=sharing) and place it in `ProjectName\Content`
3. Launch a new Conda environment, Python version 3.6, and install requirements with `pip install -r requirements.txt`
4. [Download](https://drive.google.com/file/d/19LQCuAvJIzB2I0KKTGcl0_cB0uCtcuD0/view?usp=sharing) pretrained VAE model and place it in repo's directory

## Run in Test Mode
*If you wish to reproduce results with trained model*
* Choose a map, press Play, and run `test.py`

## Run in Train Mode
*If you wish to train your own model*
* Choose a map, press Play, and open `train.py`  
  * If you are training from scratch, use `initial learning` in `train.py` and run
  * If you want to continue training model on another map, use `continual learning` in `train.py` and run

## Run in Train Mode + VAE Training 
*It's recommended to use the trained `vae.json` model, as the training time of VAE and SAC together will be long.*
1. Uncomment line `vae.optimize()` in `custom_sac.py`
2. Choose a map, press Play, use `first learn` in `train.py` and run

## Overview

This project is an experiment in deep reinforcement learning to train a self-driving (fully autonomous) race-car to drive through cones in a race track, in contribution to the Technion team in the "Formula Student Driverless" competition.

The main goal is to learn a steering policy through the cones, with a constant throttle, getting the car to a speed of about 7.5 m/s (27 km/h). 
After about an hour of training, the car will complete a lap successfully.
 
 **Pipeline**:
* Observation is obtained from simulated camera mounted on the car, cropped to remove redundant data and then encoded by VAE.
* Each VAE encoded observation is concatenated with the last 20 actions.
* Every 4 VAE encoded observation are stacked together and fed into the SAC algorithm (using google DeepMind idea).
* When the car goes out of bounds, or hits a cone, episode ends and SAC optimizations are made. 
* *If VAE optimization is enabled, VAE will also optimize.* 

![airsimPhotosFastCut](https://user-images.githubusercontent.com/38940464/109814525-4b1a6a80-7c37-11eb-9bd7-d3161d354c5c.gif)|![vaePhotosFastCut](https://user-images.githubusercontent.com/38940464/109814553-5077b500-7c37-11eb-8c4b-a14e14579726.gif)
:-------------------------:|:-------------------------:
Cropped | VAE output


**Reward function** is the distance of the car from the center of the track. The closer the car to the center, the higher the reward. If the car hits a cone of exits track, it gets a penalty. Center of the track is calculated by getting 2 closest WayPoints to the car, and the calculation of the distance between the car and the line connecting those 2 WayPoints.

 ![Screenshot from 2021-03-03 15-53-01](https://user-images.githubusercontent.com/38940464/109815678-94b78500-7c38-11eb-96d6-f9320bd783d7.png) | ![Screen Shot 2021-03-09 at 19 07 58](https://user-images.githubusercontent.com/38940464/110509482-d352b100-810a-11eb-896b-198b88fd50ba.png)
 :------------------------:|:---------------------:
 WayPoints example | Distance calculation
 
 Calculation of distance:
 
 ![equation](https://latex.codecogs.com/gif.latex?%5C%5C%20p_0%20-%20car%27s%5C%20position%20%5C%5C%20p_1%20%2C%20p_2%20-%20closest%5C%20WayPoints%20%5C%5C%20dist%20-%20distance%5C%20between%5C%20car%5C%20and%5C%20line%5C%20connecting%5C%20WayPoints%5C%20%3D%5Cfrac%7B%7C%5Cvec%7Bp_2p_1%7D%5Ctimes%5Cvec%7Bp_2p_0%7D%7C%7D%7B%7C%5Cvec%7Bp_2p_1%7D%7C%7D)
 


## Citation

If this project helped you, please cite this repository in publications:

```
@misc{Reinforcement-Learning-on-Autonomous-Race-Car,
  author = {Kanfi, Elior},
  title = {Reinforcement Learning on Autonomous Race Car},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eliork/Reinforcement-Learning-on-Autonomous-Race-Car/}},
}
```
## Credits
* Zadok, D.; Hirshberg, T.; Biran, A.; Radinsky, K. and Kapoor, A. (2019). Explorations and Lessons Learned in Building an Autonomous Formula SAE Car from Simulations.In Proceedings of the 9th International Conference on Simulation and Modeling Methodologies, Technologies and Applications - Volume 1: SIMULTECH, ISBN 978-989-758-381-0, pages 414-421. DOI: 10.5220/0008120604140421
* [AirSim](https://github.com/microsoft/AirSim)
* [AWS DeepRacer](https://aws.amazon.com/deepracer/)
* [Formula Student Technion Driverless - Based on AirSim](https://github.com/FSTDriverless/AirSim)
* [learning-to-drive-in-a-day](https://github.com/r7vme/learning-to-drive-in-a-day)
* [Learning to Drive Smoothly in Minutes](https://github.com/araffin/learning-to-drive-in-5-minutes)
* [OpenAI GYM](https://github.com/openai/gym)
* [Stable-Baselines](https://github.com/hill-a/stable-baselines)






