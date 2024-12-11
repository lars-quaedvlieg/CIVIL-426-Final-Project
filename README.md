# Context-Aware Temporal Modeling for Anomaly Detection in Hydropower Systems

This repository contains the final project for CIVIL-426, focusing on developing a context-aware temporal model to detect anomalies in hydropower systems. The project leverages sequence modeling techniques and incorporates operating mode contexts to enhance predictive accuracy.

## Repository Structure

- **`data/`**: Contains datasets used for training and evaluation.
- **`notebooks/`**: Jupyter notebooks detailing data exploration and preliminary analyses.
- **`src/`**: Source code implementing the model and associated utilities.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`Alpiq Final project 2024 1.0.pdf`**: Project report detailing methodologies and findings.
- **`requirements.txt`**: Lists Python dependencies required to run the project.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Data Preparation
Place the necessary datasets into the data/ directory. Ensure the data is preprocessed as outlined in the project report.

## Running the Model
Navigate to the src/ directory and execute the training script:

```
Copy code
python preprocess.py
python train_model.py
```

## Experiments
Navigate to the src/ directory and execute the evaluation scripts (in order)

```
python test_model_outputs.py
python test_anomaly_detection.py
```

## Project Report
For an in-depth understanding of the project's objectives, methodologies, and outcomes, consult the project report.
