import torch
import numpy as np
import matplotlib as plt

def compute_score(pred_data, data, weights=None):
    """
    Computes the score between pred_data and data for each sensor,
    and adds a weighted sum of the scores as the final column.

    Args:
        pred_data (torch.Tensor): Predicted values of size (batch_size, num_sensors).
        data (torch.Tensor): Target values of size (batch_size, num_sensors).
        weights (torch.Tensor or None): Weights for each sensor.
                                        Size: (num_sensors,). Defaults to equal weights.

    Returns:
        torch.Tensor: Tensor of size (batch_size, num_sensors + 1).
    """
    # Ensure the inputs have the same size
    assert pred_data.size() == data.size(), "pred_data and data must have the same shape."

    batch_size, num_sensors = pred_data.size()

    # Compute the score as the absolute difference (can replace with other functions)
    scores = torch.abs(pred_data - data)

    # Define default weights if none are provided (equal weights)
    if weights is None:
        weights = torch.ones(num_sensors, device=pred_data.device) / num_sensors

    # Ensure weights have the correct shape
    assert weights.size(0) == num_sensors, "weights size must match the number of sensors."

    # Compute the weighted sum of scores
    weighted_sum = (scores * weights).sum(dim=1, keepdim=True)

    # Concatenate scores and weighted sum
    result = torch.cat((scores, weighted_sum), dim=1)

    return result


def windowed_threshold_batch(scores_buffer, threshold, window_size, state_buffer,
                             nb_max_consecutive_anomalies, batch_size):
    """
    Computes anomaly states for each sensor and total score using windowed means.

    Args:
        scores_buffer (np.ndarray): Circular buffer of scores, shape (total_size, num_sensors + 1).
        threshold (float): Threshold for anomaly detection.
        window_size (int): Size of the sliding window.
        state_buffer (list): Circular buffer of states, one for each sensor + total score.
        nb_max_consecutive_anomalies (int): Number of consecutive anomalies before transitioning to state 2.
        batch_size (int): Number of new data points added to scores_buffer in this batch.

    Returns:
        list: Updated state_buffer with new states for each sensor + total score.
    """
    scores_buffer = np.array(scores_buffer)
    total_size, num_features = np.shape(scores_buffer)  # num_features = num_sensors + 1
    if total_size < window_size:
        raise ValueError("Not enough data in scores_buffer to compute window.")

    # Ensure state_buffer has the correct structure
    while len(state_buffer) < num_features:
        state_buffer.append([])

    # Process each sensor (and the total score)
    for feature_idx in range(num_features):
        start_idx = max(0, total_size - batch_size)  # Start index for new data

        for i in range(start_idx, total_size):
            if i >= window_size - 1:
                # Compute window mean for the current feature
                window_mean = np.mean(scores_buffer[i - window_size + 1:i + 1, feature_idx])

                if window_mean > threshold:
                    count = 1
                    while len(state_buffer[feature_idx]) > count and state_buffer[feature_idx][-(count + 1)] != 0:
                        count += 1

                    if count > nb_max_consecutive_anomalies:
                        state_buffer[feature_idx].append(2)
                    else:
                        state_buffer[feature_idx].append(1)
                else:
                    state_buffer[feature_idx].append(0)
            else:
                # Not enough data for the window; append a default state
                state_buffer[feature_idx].append(0)

        # Trim the state buffer for this feature to the maximum allowed size
        state_buffer[feature_idx] = state_buffer[feature_idx][-total_size:]

    return state_buffer

