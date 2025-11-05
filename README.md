<h1>A Simple Neural Network from Scratch</h1>

This project is a single-file, feedforward neural network built in Python using only the NumPy library. It is written to be as clear and educational as possible, demonstrating the core mechanics of how a neural network learns.

The network is trained to solve the classic XOR logic problem, a task that is famously impossible for a simpler linear model to solve.

<h1>Why Build This?</h1>

The goal of this project isn't to create the most efficient neural network, but to understand what's happening inside one. By building the core components from scratch, you can see exactly how data flows forward and how the error is propagated backward to make the network learn.

This code demystifies the "magic" of:

Activation Functions (Sigmoid)

Feedforward Prediction

Loss Calculation

Backpropagation (The Chain Rule)

Gradient Descent (Weight & Bias Updates)

<h1>Features</h1>

Pure Python & NumPy: No high-level libraries like TensorFlow or PyTorch.

Fully From Scratch: All core logic (feedforward, backpropagation) is implemented manually.

Sigmoid Activation: Uses the standard sigmoid activation function.

Batch Training: The train method processes the entire dataset at once (batch gradient descent).

<h1>How It Works</h1>Ωœ

The network is contained in the neural_network.py file and is built in a single class.

__init__ (Initialization): The NeuralNetwork class is initialized with a specific size (e.g., 2 inputs, 4 hidden neurons, 1 output). The weights and biases are created with small random values.

feedforward(inputs): This method takes an input (like [0, 1]) and passes it through the network, using matrix multiplication (dot products) and the sigmoid activation function to produce a final prediction between 0 and 1.

train(X, y, epochs, ...): This is the learning engine.

It loops for a set number of epochs.

In each loop, it performs a forward pass on all the data (X).

It calculates the error (the difference between its predictions and the true answers y).

It performs a backward pass (backpropagation) to calculate the "gradient" (the direction of error) for every weight and bias.

It updates all weights and biases, "nudging" them in the right direction to reduce the error.

<h1>How to Run</h1>

<h2>Prerequisites</h2>

Python 3.x

NumPy

<h2>1. Install NumPy</h2>

If you don't have NumPy installed, you can get it via pip:

pip install numpy


<h2>2. Run the Script</h2>

Save the code as neural_network.py and run it from your terminal:

python neural_network.py


<h2>3. Expected Output</h2>

You will first see the network's structure, then see the training progress as the loss (error) decreases every 1000 epochs. Finally, you'll see the network's predictions on all four XOR cases.

--- Initializing Neural Network ---
  Input Layer Size: 2
  Hidden Layer Size: 4
  Output Layer Size: 1

--- Starting Training ---
Epochs: 10000, Learning Rate: 0.1
Epoch 1000/10000, Loss: 0.493863
Epoch 2000/10000, Loss: 0.457855
Epoch 3000/10000, Loss: 0.288544
Epoch 4000/10000, Loss: 0.091309
Epoch 5000/10000, Loss: 0.046556
Epoch 6000/10000, Loss: 0.030999
Epoch 7000/10000, Loss: 0.023249
Epoch 8000/10000, Loss: 0.018596
Epoch 9000/10000, Loss: 0.015525
Epoch 10000/10000, Loss: 0.013348
--- Training Complete ---

--- Testing the Network After Training ---
Input: [0 0], Prediction: 0.0125, Rounded: 0
Input: [0 1], Prediction: 0.9892, Rounded: 1
Input: [1 0], Prediction: 0.9892, Rounded: 1
Input: [1 1], Prediction: 0.0104, Rounded: 0


(Note: Your exact prediction values may differ slightly due to the random initialization of weights.)

<h1>License</h1>

This project is open-source and available under the MIT License.
