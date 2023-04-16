import torch
import sys
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.dataset import COCODataset
from models.detector_network import DetectorNetwork
from models.reinforcement_learning_agent import ReinforcementLearningAgent
from utils.utils import compute_losses
from utils.evaluator import Evaluator
from collections import defaultdict
from utils.active_learning_stategies import estimate_uncertainty, select_samples_for_annotation

class Trainer:
    """
    A class for training the detector network and reinforcement learning agent.
    """

    def __init__(self, config, detector_network, reinforcement_agent, train_dataset, val_dataset):
        """
        Initializes the Trainer class.

        Args:
            config (dict): The configuration dictionary for the training settings.
            dataset (COCODataset): The dataset to be used for training.
            detector_network (DetectorNetwork): The detector network to be trained.
            reinforcement_learning_agent (ReinforcementLearningAgent): The reinforcement learning agent to be trained.
            device (torch.device): The device to be used for training (e.g., 'cpu' or 'cuda').
        """
        self.config = config
        self.detector_network = detector_network
        self.reinforcement_agent = reinforcement_agent
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config['train_batch_size'], shuffle=True, num_workers=self.config['num_workers'])
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config['val_batch_size'], shuffle=False, num_workers=self.config['num_workers'])

    def train(self):
        # Set the detector network and reinforcement agent to training mode
        self.detector_network.train()

        # Set up the optimizer
        optimizer = torch.optim.Adam(
            self.detector_network.parameters(), lr=self.config['learning_rate'])

        for epoch in range(self.config['num_epochs']):
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")

            # Initialize the running losses
            running_losses = defaultdict(float)

            # Iterate through the training dataset
            for images, targets in self.train_loader:
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]

                # Forward pass through the detector network
                detector_output = self.detector_network(images)

                # Calculate the object detection losses
                detection_losses = compute_losses(detector_output, targets)

                # Estimate the uncertainty of the detector network's predictions
                uncertainties = estimate_uncertainty(detector_output)

                # Forward pass through the reinforcement agent
                agent_output = self.reinforcement_agent(
                    images, detector_output)

                # Select the samples for annotation
                selected_samples = select_samples_for_annotation(
                    uncertainties, agent_output)
                # Train the reinforcement learning agent
                self.reinforcement_agent.train(
                    images, detector_output, targets, self.config['rl_agent']['num_timesteps'])

                # Forward pass through the reinforcement agent
                agent_output = self.reinforcement_agent(
                    images, detector_output)

                # Calculate the reinforcement learning losses
                rl_losses = self.reinforcement_agent.compute_losses(
                    agent_output, targets)

                # Combine the object detection losses and reinforcement learning losses
                total_losses = {
                    key: detection_losses[key] + rl_losses[key] for key in detection_losses.keys()}

                # Reset the gradients
                optimizer.zero_grad()

                # Backward pass
                sum(total_losses.values()).backward()

                # Optimization step
                optimizer.step()

                # Update the running losses
                for key in total_losses.keys():
                    running_losses[key] += total_losses[key].item()

            # Compute the average losses for this epoch
            epoch_losses = {key: running_loss / len(self.train_loader)
                            for key, running_loss in running_losses.items()}
            print(f"Training losses: {epoch_losses}")

            # Evaluate the model on the validation dataset
            evaluator = Evaluator(config=self.config, detector_network=self.detector_network,
                                  reinforcement_agent=self.reinforcement_agent, val_dataset=self.val_dataset, device=self.device)
            val_metrics = evaluator.evaluate()

            print(f"Validation metrics: {val_metrics}")
