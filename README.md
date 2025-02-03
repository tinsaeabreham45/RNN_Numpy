# First RNN Implementation Using NumPy

This repository contains an implementation of a simple Recurrent Neural Network (RNN) using only NumPy. The goal is to understand the fundamental concepts of RNNs by manually implementing the forward propagation of an RNN cell and a full unrolled RNN across multiple time steps.

## Features
- **`rnn_cell_forward(xt, a_prev, parameters)`**: Computes the forward pass for a single RNN cell.
- **`rnn_forward(x, a0, parameters)`**: Implements the full forward propagation through time for an RNN.

## Implementation Details
- Uses **only NumPy** (no TensorFlow or PyTorch).
- Supports multiple time steps and batch processing.
- Applies the **tanh** activation function for hidden states.
- Outputs predictions using the **softmax** function.

## Purpose
This notebook serves as a foundational step toward understanding how RNNs work internally before moving on to more advanced deep learning frameworks like TensorFlow or PyTorch.

