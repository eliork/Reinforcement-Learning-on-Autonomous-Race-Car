import os
from custom_sac import SACWithVAE
from stable_baselines.common.vec_env import VecFrameStack,  DummyVecEnv
from gym_airsim.airsim_car_env import AirSimCarEnv
from vae.controller import VAEController
import numpy as np
import cv2


PATH_MODEL_SAC = "sac.zip"
PATH_MODEL_VAE = "vae.json"

vae = VAEController()
airsim_env = lambda: AirSimCarEnv(vae)
env = DummyVecEnv([airsim_env])
env = VecFrameStack(env,4)


# Run in test mode if trained models exist.
if os.path.exists(PATH_MODEL_SAC) and os.path.exists(PATH_MODEL_VAE):
    print("Task: test")
    sac = SACWithVAE.load(PATH_MODEL_SAC, env)
    vae.load(PATH_MODEL_VAE)

    obs = env.reset()
    while True:
        arr = vae.decode(obs[:,:, :512].reshape(1, 512))
        arr = np.round(arr).astype(np.uint8)
        arr = arr.reshape(80, 160, 3)
        # to visualize what car sees
        #cv2.imwrite("decoded_img.png", arr)
        action, _states = sac.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
else:
    print('models does not exist')