import torch
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader
import numpy as np


class Evaluator:
    def __init__(self, config, detector_network, reinforcement_agent, val_dataset):
        self.config = config
        self.detector_network = detector_network
        self.reinforcement_agent = reinforcement_agent
        self.val_dataset = val_dataset
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config['val_batch_size'], shuffle=False, num_workers=self.config['num_workers'])

    def evaluate(self):
        # Set the detector network and reinforcement agent to evaluation mode
        self.detector_network.eval()
        self.reinforcement_agent.eval()

        # Initialize evaluation metrics
        total_correct = 0
        total_objects = 0

        with torch.no_grad():
            # Iterate through the validation dataset
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]

                # Forward pass through the detector network
                output = self.detector_network(images)

                # Evaluate the predictions
                num_correct, num_objects = self.evaluate_predictions(
                    output, targets)

                total_correct += num_correct
                total_objects += num_objects

        # Calculate evaluation metrics
        precision = total_correct / total_objects
        return {'precision': precision}

    def evaluate_predictions(self, output, targets):
        iou_threshold = 0.5
        total_correct = 0
        total_objects = 0

        for pred_boxes, target_boxes in zip(output, targets):
            # Convert the predicted and target boxes to the format: (x1, y1, x2, y2)
            pred_boxes = pred_boxes[:, :4]
            target_boxes = target_boxes[:, :4]

            # Calculate the total number of objects
            total_objects += target_boxes.shape[0]

            # Initialize a list of matched target boxes
            matched_targets = []

            # Iterate through the predicted boxes
            for pred_box in pred_boxes:
                # Calculate IoU with each target box
                ious = [self.iou(pred_box, target_box)
                        for target_box in target_boxes]

                # Find the index of the maximum IoU value
                max_iou_idx = np.argmax(ious)

                # Check if the maximum IoU is greater than the threshold and the target box is not matched already
                if ious[max_iou_idx] >= iou_threshold and max_iou_idx not in matched_targets:
                    total_correct += 1
                    matched_targets.append(max_iou_idx)

        return total_correct, total_objects


        def iou(self, box1, box2):
            x1, y1, x2, y2 = box1
            x1_gt, y1_gt, x2_gt, y2_gt = box2

            xi1 = max(x1, x1_gt)
            yi1 = max(y1, y1_gt)
            xi2 = min(x2, x2_gt)
            yi2 = min(y2, y2_gt)

            intersection_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

            union_area = box1_area + box2_area - intersection_area

            return intersection_area / union_area
