def compute_reward(detection_performance, annotation_cost):
    """
    Computes the reward based on the detection performance and annotation cost.
    
    Args:
        detection_performance (float): The detection performance metric (e.g., mAP).
        annotation_cost (float): The cost associated with annotating the selected samples.
        
    Returns:
        float: The computed reward.
    """