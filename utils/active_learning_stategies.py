import numpy as np

def estimate_uncertainty(predictions):
    """
    Estimates the uncertainty of the detector network's predictions (e.g., using entropy or variance).
    
    Args:
        predictions (array-like): The output predictions of the detector network.
        
    Returns:
        array-like: The estimated uncertainty for each prediction.
    """
    # Convert the predictions to probabilities using the softmax function
    probabilities = np.exp(predictions) / \
        np.sum(np.exp(predictions), axis=-1, keepdims=True)

    # Calculate the entropy for each predicted class probability distribution
    entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)

    return entropy


def select_samples_for_annotation(uncertainties, agent_action):
    """
    Selects informative and diverse samples for annotation based on the uncertainty estimation and the reinforcement learning agent's action.
    
    Args:
        uncertainties (array-like): The estimated uncertainties for each prediction.
        agent_action (int): The action selected by the reinforcement learning agent.
        
    Returns:
        list: The indices of the selected samples for annotation.
    """
    # Determine the number of samples to select based on the agent's action
    num_samples_to_select = agent_action

    # Get the indices of the top-k most uncertain samples
    selected_indices = np.argpartition(
        uncertainties, -num_samples_to_select)[-num_samples_to_select:]

    return list(selected_indices)
