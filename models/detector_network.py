import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn


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

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas