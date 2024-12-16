# RNN Implementation
Assignment 7 from CS 4445 Datamining

# Homework Assignment: RNN Implementation

This repository contains solutions for two assignments focused on implementing and testing a custom Recurrent Neural Network (RNN) for binary sequence classification. Each problem involves a set of specific tasks to understand and build fundamental RNN components using PyTorch. Below is a detailed description of the two problems.

---

## Problem 1: Implementing RNN Components

### Objective:
To implement key components of an RNN class from scratch, focusing on the computation of intermediate values and hidden states at each time step.

### Tasks:
1. **Initialize Parameters**:
   - Define and initialize the following trainable parameters:
     - Weight matrices: `U`, `W`, `V`
     - Bias terms: `b_h`, `b`
   - Ensure parameters are properly sized to handle arbitrary input dimensions and sequence lengths.

2. **Compute Logits (zt)**:
   - Implement the `compute_zt` method to calculate the linear logits `zt` for the RNN layer at a specific time step.
   - Equation:
     \[ z_t = Wx_t + Uh_{t-1} + b_h \]

3. **Compute Hidden State (ht)**:
   - Implement the `compute_ht` method to compute the hidden state `ht` using the Tanh activation function.
   - Equation:
     \[ h_t = \tanh(z_t) \]

4. **RNN Step Function**:
   - Develop the `step` method to process an input vector at a single time step, updating the hidden state.
   - This function should:
     - Use `compute_zt` to calculate logits.
     - Use `compute_ht` to update the hidden state.

### Testing:
Each method is tested using `pytest`. Run the following commands to verify the implementation:
- **`compute_zt`**: `pytest -v test_1.py -m RNN_compute_zt`
- **`compute_ht`**: `pytest -v test_1.py -m RNN_compute_ht`

---

## Problem 2: Full RNN Implementation for Sequence Classification

### Objective:
To complete the RNN implementation and train it to classify binary sequences.

### Tasks:
1. **Sequence Processing**:
   - Extend the `RNN` class to process a sequence of inputs over multiple time steps using the `step` method iteratively.

2. **Output Prediction**:
   - Implement a fully connected layer to compute the final output prediction based on the hidden state of the last time step.
   - Equation:
     \[ y = \sigma(Vh_T + b) \]
     where \( \sigma \) is the sigmoid activation function.

3. **Training the RNN**:
   - Use the stochastic gradient descent (SGD) optimizer to update parameters.
   - Train the RNN on binary sequence data to classify each sequence into one of two classes (e.g., wake word detection).

4. **Evaluation**:
   - Evaluate the RNN’s performance on test data, ensuring it correctly processes variable-length sequences and makes accurate predictions.

### Testing:
Unit tests are provided to validate the implementation of sequence processing and output prediction. Use the following commands:
- **Sequence Processing**: `pytest -v test_2.py -m RNN_sequence_processing`
- **Output Prediction**: `pytest -v test_2.py -m RNN_output_prediction`
