import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn,  FastRCNNPredictor


class DetectorNetwork(nn.Module):
    """
    A class for the detector network using the Faster R-CNN model.
    """

    def __init__(self, config):
        """
        Initializes the DetectorNetwork class.

        Args:
            config (dict): The configuration dictionary for the network settings.
        """
        super(DetectorNetwork, self).__init__()
        self.config = config
        self.model = self._create_model()

    def _create_model(self):
        """
        Creates the Faster R-CNN model with a custom number of object classes.

        Returns:
            model (torchvision.models.detection.FasterRCNN): The Faster R-CNN model.
        """
        model = fasterrcnn_resnet50_fpn(pretrained=self.config['pretrained'])
        # +1 for background class
        num_classes = len(self.config['class_labels']) + 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        return model

    def forward(self, images):
        """
        Performs the forward pass of the detector network.

        Args:
            images (torch.Tensor): A tensor containing the input images.

        Returns:
            detections (List[Dict[Tensor]]): A list of dictionaries containing the detection results.
        """
        detections = self.model(images)
        return detections
