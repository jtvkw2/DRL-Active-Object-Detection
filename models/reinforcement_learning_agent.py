import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from .reinforcement_learning_env import ObjectDetectionEnv


class ReinforcementLearningAgent(nn.Module):
    """
    A class for the reinforcement learning agent.
    """

    def __init__(self, config):
        super().__init__()
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

    def train(self, images, detector_output, targets, num_timesteps):
        """
        Trains the reinforcement learning agent.

        Args:
            images (tensor): The input images.
            detector_output (list): The output from the initial object detector.
            targets (list): The ground truth bounding box targets.
            num_timesteps (int): The number of timesteps to train the agent.
        """
        for _ in range(num_timesteps):
            # Get the agent's output
            agent_output = self.forward(images, detector_output)

            # Compute the reward
            reward = self._reward_function(agent_output, targets)

            # Perform the agent update based on the reward
            # You might need to modify this part depending on your chosen RL library
            self.agent.update(reward)

    def forward(self, images, detector_output):
        """
        Forward pass through the reinforcement learning agent.

        Args:
            images (torch.Tensor): The input images.
            detector_output (dict): The output from the detector network.

        Returns:
            agent_output (dict): The output from the reinforcement learning agent.
        """
        # Extract the batch size from the images tensor
        batch_size = images.size(0)

        # Create the observations for the agent
        observations = self._create_observations(images, detector_output)

        # Get the actions from the agent
        actions, _ = self.agent.predict(observations, deterministic=True)

        # Reshape the actions to match the batch size
        actions = torch.from_numpy(actions).view(batch_size, 1)

        # Apply the actions to the detector output
        adjusted_scores = detector_output['scores'] * actions.to(images.device)

        # Create the agent_output dictionary
        agent_output = {'adjusted_scores': adjusted_scores}

        return agent_output


    def compute_losses(self, agent_output, targets):
        """
        Computes the reinforcement learning losses.

        Args:
            agent_output (dict): The output from the reinforcement learning agent.
            targets (list): The ground truth bounding box targets.

        Returns:
            rl_losses (dict): The reinforcement learning losses.
        """
        # Compute the reinforcement learning loss based on the agent's design.
        # In this example, we use the difference between the agent's adjusted
        # scores and the ground truth labels as the loss.
        rl_loss = nn.functional.mse_loss(agent_output['adjusted_scores'], targets)

        # Create the rl_losses dictionary
        rl_losses = {'rl_loss': rl_loss}

        return rl_losses

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
    
    def _create_observations(self, images, detector_output):
        """
        Creates the observations for the reinforcement learning agent.

        Args:
            images (torch.Tensor): The input images.
            detector_output (dict): The output from the detector network.

        Returns:
            observations (np.ndarray): The observations for the reinforcement learning agent.
        """
        # Implement the logic to create the observations.
        # In this example, we concatenate the image features and detection scores as the observations.
        image_features = images.view(images.size(0), -1).numpy()
        detection_scores = detector_output['scores'].view(-1, 1).cpu().numpy()

        observations = np.concatenate(
            (image_features, detection_scores), axis=1)

        return observations
    
    def _reward_function(self, agent_output, targets):
        """
        Compute the reward for the reinforcement learning agent.

        Args:
            agent_output (dict): The output from the reinforcement learning agent.
            targets (list): The ground truth bounding box targets.

        Returns:
            reward (float): The reward for the agent.
        """
        # Implement the reward computation logic.
        # In this example, we consider the mean Intersection over Union (IoU) as the reward.
        adjusted_boxes = agent_output['adjusted_boxes']
        reward = 0
        num_targets = len(targets)

        for i in range(num_targets):
            iou = self._compute_iou(adjusted_boxes[i], targets[i])
            reward += iou

        reward /= num_targets
        return reward

