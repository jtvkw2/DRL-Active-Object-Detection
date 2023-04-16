import torch
from utils.dataset import COCODataset
from models.detector_network import DetectorNetwork
from models.reinforcement_learning_agent import ReinforcementLearningAgent
from utils.trainer import Trainer
from config.config import config


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    train_dataset = COCODataset(root_dir=config['root_dir'], set_name='train',
                                image_dir=config['train_img_dir'], annotation_path=config['train_annotation_dir'], transform=None)

    val_dataset = COCODataset(root_dir=config['root_dir'], set_name='val',
                              image_dir=config['val_img_dir'], annotation_path=config['val_annotation_dir'], transform=None)

    # Initialize the detector network
    detector_network = DetectorNetwork(config).to(device)

    # Initialize the reinforcement learning agent
    reinforcement_agent = ReinforcementLearningAgent(config)

    # Train the model
    trainer = Trainer(config=config, detector_network=detector_network,
                      reinforcement_agent=reinforcement_agent, train_dataset=train_dataset, val_dataset=val_dataset)
    trainer.train()


if __name__ == '__main__':
    main()
