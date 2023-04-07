import torch
import yaml
from utils.dataset import COCODataset
from models.detector_network import DetectorNetwork
from models.reinforcement_learning_agent import ReinforcementLearningAgent
from utils.trainer import Trainer
from utils.evaluator import Evaluator


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    dataset = COCODataset(config['data_dir'], set_name=config['set_name'])

    # Initialize the detector network
    detector_network = DetectorNetwork(config).to(device)

    # Initialize the reinforcement learning agent
    reinforcement_learning_agent = ReinforcementLearningAgent(config)

    # Train the model
    trainer = Trainer(config, dataset, detector_network,
                      reinforcement_learning_agent, device)
    for epoch in range(config['num_epochs']):
        trainer.train(epoch)

    # Evaluate the model
    evaluator = Evaluator(config, dataset, detector_network, device)
    evaluation_results = evaluator.evaluate()

    print("Evaluation Results:", evaluation_results)


if __name__ == '__main__':
    main()
