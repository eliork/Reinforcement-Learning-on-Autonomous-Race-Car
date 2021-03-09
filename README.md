# Reinforcement Learning on Autonomous Race Car

Reinforcement Learning approach to "Formula Student Technion Driverless" project, simulated in Unreal Engine 4 with AirSim plugin, using Soft Actor Critic (SAC) algorithm and Variational Auto Encoder (VAE).

![gifToGit](https://user-images.githubusercontent.com/38940464/109778141-26a99880-7c0d-11eb-801d-807bf91cded4.gif)

## Prerequisites 
* Operating System: Ubuntu 18.04 or Windows 10
* Software: Unreal Engine 4.24.3
* GPU: Nvidia GTX 1080 or higher (recommended)
  
## How To Build
1. Set up Unreal Engine 4, AirSim and the FSTD Environment by following this link [Ubuntu](https://github.com/eliork/Reinforcement-Learning-Approach-to-Autonomous-Race-Car2/blob/main/build_FSTDriverless_ubuntu.md) , [Windows](https://github.com/FSTDriverless/AirSim/blob/master/docs/build_FSTDriverless_windows.md)
  2. If you are using Ubuntu, you can skip this step. If you are using Windows: Download updated RaceCourse folder from this [link](https://drive.google.com/file/d/1lpBHRYYw7GRICgLaSfMQcXlbP2A98b9L/view?usp=sharing) and place it in `ProjectName\Content`
3. Launch a new Conda environment, Python version 3.6, and install requirements with `pip install -r requirements.txt`

## Run in Test Mode
*If you wish to reproduce results with trained model*
* Choose a map, press Play, and run `test.py`

## Run in Train Mode
*If you wish to train your own model*
* Choose a map, press Play, and open `train.py`  
  * If you are training from scratch, use `first learn` in `train.py` and run
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

 ![Screenshot from 2021-03-03 15-53-01](https://user-images.githubusercontent.com/38940464/109815678-94b78500-7c38-11eb-96d6-f9320bd783d7.png)
 :------------------------:
 WayPoints example


## Citation

If this project helped you, please cite this repository in publications:

```
@misc{Reinforcement-Learning-Approach-to-Autonomous-Race-Car,
  author = {Kanfi, Elior},
  title = {Reinforcment Learning Approach to Autonomous Race Car},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eliork/Reinforcement-Learning-Approach-to-Autonomous-Race-Car/}},
}
```
## Credits
* [Formula Student Technion Driverless - Based on AirSim](https://github.com/FSTDriverless/AirSim)
* [AirSim](https://github.com/microsoft/AirSim)
* [Learning to Drive Smoothly in Minutes](https://github.com/araffin/learning-to-drive-in-5-minutes)
* [learning-to-drive-in-a-day](https://github.com/r7vme/learning-to-drive-in-a-day)
* [Stable-Baselines](https://github.com/hill-a/stable-baselines)
* [OpenAI GYM](https://github.com/openai/gym)
* [AWS DeepRacer](https://aws.amazon.com/deepracer/)





