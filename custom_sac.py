import time
from collections import deque

import numpy as np
from stable_baselines import SAC
from stable_baselines import logger
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.math_util import safe_mean
from stable_baselines.common import TensorboardWriter


import cv2



class SACWithVAE(SAC):
    """
    Custom version of Soft Actor-Critic (SAC) to use it with donkey car env.
    It is adapted from the stable-baselines version.

    Notable changes:
    - optimization is done after each episode and not at every step

    """
    def optimize(self, step, writer, current_lr):
        """
        Do several optimization steps to update the different networks.

        :param step: (int) current timestep
        :param writer: (TensorboardWriter object)
        :param current_lr: (float) Current learning rate
        :return: ([np.ndarray]) values used for monitoring
        """
        train_start = time.time()
        mb_infos_vals = []
        for grad_step in range(self.gradient_steps):
            if step < self.batch_size or step < self.learning_starts:
                break
            self.n_updates += 1
            # Update policy and critics (q functions)
            mb_infos_vals.append(self._train_step(step, writer, current_lr))

            if (step + grad_step) % self.target_update_interval == 0:
                # Update target network
                self.sess.run(self.target_update_op)
        if self.n_updates > 0:
            print("SAC training duration: {:.2f}s".format(time.time() - train_start))
        return mb_infos_vals

    def learn(self, total_timesteps, callback=None,
              log_interval=1, tb_log_name="SAC", print_freq=100,vae=None):

        self.learning_rate = get_schedule_fn(self.learning_rate)

        callback = self._init_callback(callback)

        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)

            start_time = time.time()
            episode_rewards = [0.0]

            obs = self.env.reset()

            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            ep_len = 0
            self.n_updates = 0
            infos_values = []
            mb_infos_vals = []
            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            for step in range(total_timesteps):
                # Compute current learning_rate
                frac = 1.0 - step / total_timesteps
                current_lr = self.learning_rate(frac)

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback.on_step() is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if step < self.learning_starts:
                    action = self.env.action_space.sample()
                    # No need to rescale when sampling random action
                    rescaled_action = action
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)
                ep_len += 1
                callback.update_locals(locals())
                if callback.on_step() is False:
                    break

                ##################
                arr = vae.decode(new_obs[:, :512].reshape(1, 512))
                arr = np.round(arr).astype(np.uint8)
                arr = arr.reshape(80, 160, 3)
                #to visualize what car sees
                #cv2.imwrite("decoded_img.png", arr)

                ###############3
                if print_freq > 0 and ep_len % print_freq == 0 and ep_len > 0:
                    print("{} steps".format(ep_len))

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, step)

                if ep_len > self.train_freq:
                    print("Additional training")
                    self.env.reset()
                    mb_infos_vals = self.optimize(step, writer, current_lr)
                    done = True


                episode_rewards[-1] += reward
                if done:
                    obs = self.env.reset()
                    print("Episode finished. Reward: {:.2f} {} Steps".format(episode_rewards[-1], ep_len))
                    episode_rewards.append(0.0)
                    ep_len = 0
                    mb_infos_vals = self.optimize(step, writer, current_lr)



                    # train VAE
                    train_start = time.time()
                    #training VAE with SAC
                    #vae.optimize()
                    print("VAE training duration:", time.time() - train_start)
                    obs = self.env.reset()

                callback.on_rollout_end()

                # Log losses and entropy, useful for monitor training
                if len(mb_infos_vals) > 0:
                    infos_values = np.mean(mb_infos_vals, axis=0)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                callback.on_rollout_start()

                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                    logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", self.n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', "{:.2f}".format(time.time() - start_time))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", step)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []




            # Use last batch
            print("Final optimization before saving")
            self.env.reset()
            mb_infos_vals = self.optimize(step, writer, current_lr)
        callback.on_training_end()

        return self
