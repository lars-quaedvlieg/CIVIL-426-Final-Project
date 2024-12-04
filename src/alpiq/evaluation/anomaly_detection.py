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


def windowed_threshold_batch(scores_buffer, threshold, window_size, state_buffer, nb_max_consecutive_anomalies, batch_size):
    """
    Computes the state for the last #batch_size data points in scores_buffer
    using a sliding window mean, and transitions states based on thresholds.

    Args:
        scores_buffer (np.ndarray): Circular buffer of scores.
        threshold (float): Threshold for anomaly detection.
        window_size (int): Size of the sliding window.
        state_buffer (list): Circular buffer of states.
        nb_max_consecutive_anomalies (int): Number of consecutive anomalies before transitioning to state 2.
        batch_size (int): Number of new data points added to scores_buffer in this batch.

    Returns:
        list: Updated state_buffer with new states included.
    """
    # Ensure enough data for window computation
    buffer_size = len(scores_buffer)
    if buffer_size < window_size:
        raise ValueError("Not enough data in scores_buffer to compute window.")

    # Start index for new data
    start_idx = max(0, buffer_size - batch_size)

    for i in range(start_idx, buffer_size):
        # Only compute if there's enough data for the window
        if i >= window_size - 1:
            # Compute the mean for the current window
            window_mean = np.mean(scores_buffer[i - window_size + 1: i + 1])

            if window_mean > threshold:
                count = 0
                # Count consecutive anomalies in the state buffer
                while len(state_buffer) > count and state_buffer[-(count + 1)] != 0:
                    count += 1

                if count > nb_max_consecutive_anomalies:
                    state_buffer.append(2)
                else:
                    state_buffer.append(1)
            else:
                state_buffer.append(0)
        else:
            # Not enough data for the window; append a default state
            state_buffer.append(0)

    # Keep the state buffer within the maximum buffer size
    state_buffer = state_buffer[-buffer_size:]

    return state_buffer
