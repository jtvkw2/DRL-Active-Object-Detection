def intersection_over_union(boxes_a, boxes_b):
    """
    Computes the intersection over union (IoU) between two sets of bounding boxes.
    
    Args:
        boxes_a (torch.Tensor): A tensor of shape (N, 4) representing the first set of bounding boxes.
        boxes_b (torch.Tensor): A tensor of shape (M, 4) representing the second set of bounding boxes.
        
    Returns:
        torch.Tensor: A tensor of shape (N, M) representing the IoU between each pair of bounding boxes in boxes_a and boxes_b.
    """


def nms(boxes, scores, threshold):
    """
    Applies non-maximum suppression (NMS) to a set of bounding boxes based on their scores and a given IoU threshold.
    
    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) representing the bounding boxes.
        scores (torch.Tensor): A tensor of shape (N,) representing the scores for each bounding box.
        threshold (float): The IoU threshold for NMS.
        
    Returns:
        torch.Tensor: A tensor of shape (K, 4) representing the bounding boxes after NMS, where K is the number of remaining boxes.
    """


def map_score(ground_truth_boxes, predicted_boxes, iou_threshold):
    """
    Computes the mean average precision (mAP) score for a set of predicted bounding boxes and their corresponding ground truth boxes.
    
    Args:
        ground_truth_boxes (list): A list of tensors representing the ground truth bounding boxes for each image.
        predicted_boxes (list): A list of tensors representing the predicted bounding boxes for each image.
        iou_threshold (float): The IoU threshold for considering a predicted box as a true positive.
        
    Returns:
        float: The computed mAP score.
    """


def load_dataset(dataset_path, split):
    """
    Loads the dataset from the given path and split (e.g., train, val, or test).
    
    Args:
        dataset_path (str): The path to the dataset directory.
        split (str): The dataset split to load (e.g., 'train', 'val', or 'test').
        
    Returns:
        Dataset: An instance of a dataset class (e.g., COCO or PASCAL VOC) for the specified split.
    """
