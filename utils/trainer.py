import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.dataset import COCODataset
from models.detector_network import DetectorNetwork
from models.reinforcement_learning_agent import ReinforcementLearningAgent
from utils import compute_losses


class Trainer:
    """
    A class for training the detector network and reinforcement learning agent.
    """

    def __init__(self, config, dataset, detector_network, reinforcement_learning_agent, device):
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
        self.dataset = dataset
        self.detector_network = detector_network
        self.reinforcement_learning_agent = reinforcement_learning_agent
        self.device = device

        self.train_loader = DataLoader(dataset.train_data,
                                       batch_size=config["batch_size"],
                                       shuffle=True,
                                       num_workers=config["num_workers"],
                                       collate_fn=dataset.collate_fn)

        self.optimizer = Adam(self.detector_network.parameters(),
                              lr=config["learning_rate"],
                              weight_decay=config["weight_decay"])

    def train(self, epoch):
        """
        Trains the detector network and reinforcement learning agent for one epoch.

        Args:
            epoch (int): The current epoch number.
        """
        self.detector_network.train()
        self.reinforcement_learning_agent.train()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]

            # Forward pass
            predictions = self.detector_network(images)
            loss, loss_dict = compute_losses(predictions, targets, self.device)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the reinforcement learning agent
            self.reinforcement_learning_agent.update(batch_idx, loss_dict)

            if batch_idx % self.config["log_interval"] == 0:
                print(
                    f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item()}")
