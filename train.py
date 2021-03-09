from gym_airsim.airsim_car_env import AirSimCarEnv
from stable_baselines.common.vec_env import VecFrameStack,  DummyVecEnv
import numpy as np
from pathlib import Path
import os
from os.path import exists
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from vae.controller import VAEController
import cv2
from stable_baselines.sac.policies import *
from stable_baselines import *
from custom_sac import SACWithVAE


config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

vae = VAEController()
airsim_env = lambda: AirSimCarEnv(vae)
env = DummyVecEnv([airsim_env])
env = VecFrameStack(env,4)

np.random.seed(123)

log_dir = 'logs'
if not exists(log_dir):
    os.makedirs(log_dir)
tensorboard_dir = Path("logs/sac_AirsimCar_tensorboard/")
if not exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

PATH_MODEL_VAE = "vae.json"
PATH_MODEL_SAC = "sac"

# Run in training mode.
print("Task: train")

####################################

# continual learning
# use this for continuing training on a model (on a different map e.g.)

vae.load(PATH_MODEL_VAE)
airsim_env = lambda: AirSimCarEnv(vae)
env = DummyVecEnv([airsim_env])
env = VecFrameStack(env, 4)
# load your last trained model
model = SACWithVAE.load("sac.zip",env)
model.set_env(env)
model.learn(total_timesteps=1000000, vae=vae)

# save your new model
model.save("sac.zip")


#################################

# first learn
# use this for training from scratch.

'''
policy_kwargs = dict(act_fun=tf.nn.elu, layers=[256,256])
sac = SACWithVAE(MlpPolicy,
              env,
              learning_rate=0.00073,
              verbose=1,
              batch_size=256,
              gamma=0.99,
              tau=0.02,
              tensorboard_log=tensorboard_dir,
              buffer_size=300000,
              train_freq=999999,
              gradient_steps=64,
              learning_starts=500,
              ent_coef='auto',
              policy_kwargs=policy_kwargs
            )
vae.load(PATH_MODEL_VAE)
sac.learn(total_timesteps=2000000, vae=vae)
sac.save(PATH_MODEL_SAC)
'''



