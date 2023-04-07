import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from reinforcement_learning_env import ObjectDetectionEnv


class ReinforcementLearningAgent:
    """
    A class for the reinforcement learning agent.
    """

    def __init__(self, config):
        """
        Initializes the ReinforcementLearningAgent class.

        Args:
            config (dict): The configuration dictionary for the agent settings.
        """
        self.config = config
        self.env = self._create_environment()
        self.agent = self._create_agent()

    def _create_environment(self):
        """
        Creates the custom object detection environment.

        Returns:
            env (DummyVecEnv): The custom object detection environment wrapped in a DummyVecEnv.
        """
        env = ObjectDetectionEnv(self.config)
        check_env(env)
        return DummyVecEnv([lambda: env])

    def _create_agent(self):
        """
        Creates the Proximal Policy Optimization (PPO) agent.

        Returns:
            agent (stable_baselines3.PPO): The PPO agent.
        """
        agent = PPO("MlpPolicy", self.env, verbose=1)
        return agent

    def train(self, num_timesteps):
        """
        Trains the reinforcement learning agent.

        Args:
            num_timesteps (int): The number of timesteps to train the agent.
        """
        self.agent.learn(total_timesteps=num_timesteps)

    def select_action(self, observation):
        """
        Selects an action based on the current observation.

        Args:
            observation (np.ndarray): The current observation.

        Returns:
            action (np.ndarray): The selected action.
        """
        action, _ = self.agent.predict(observation, deterministic=True)
        return action
