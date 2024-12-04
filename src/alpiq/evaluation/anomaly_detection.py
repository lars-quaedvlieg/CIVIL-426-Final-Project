import torch
import numpy as np
import matplotlib as plt

def predict_single(model, data):
    pred = model(data)
    return pred

def compute_score(pred_data, data):

    offset = data - pred_data
    if offset>0:
        score = offset
    else:
        score = 0

    return score

def windowed_threshold(score, threshold, window_size, state_buffer):

    window_mean = np.mean(score[-window_size:])

    if window_mean > threshold:

        count = 0
        while state_buffer[-(count + 1)] != 0:
            count += 1

        if count > nb_:
            return 2
        else:
            return 1
    else:
        return 0 
    

def main():

    score_buffer = []
    state_buffer = []
    max_len_buffer = 100

    threshold_value = 0.9
    window_size = 10

    # Load data
    
    for batch in data_loader:
        
        #predict_timestep
        pred = predict_single(model, )
        score = compute_score(pred, batch["next_value"])

        #store score in buffer
        score_buffer.append(score)

        if len(score_buffer) > 100:
            score_buffer.pop(0)

        # if enough data in buffer
        if len(score_buffer) > 10:
            state = windowed_threshold(score_buffer, threshold_value, window_size)

            state_buffer.append(state)
