import numpy as np

def sigmoid(x):
    """
    This is our sigmoid activation function. 
    The formula is 1 / (1 + e^(-z))

    x is what our parameter is. It is the value that is put in the activation function to return an output for the node.
    It "squashes" any value to be between 0 and 1.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    """
    This is the derivative of the sigmoid function.
    It tells us the "slope" of the sigmoid curve at a point.
    
    We use this in backpropagation to know how much to adjust our weights.
    
    This formula is a clever optimization. It assumes 'x' is *already* the output of a sigmoid function (which it will be when we use this). 
    The full formula is: sigmoid(x) * (1 - sigmoid(x))
    But since our input 'x' will already be sigmoid(x), we just
    do x * (1 - x).
    """
    return x * (1 - x)

class neuralNetwork:
    """
    This class will hold our entire neural network.
    - It will store the weights and biases.
    - It will have a 'feedforward' method to make predictions.
    - It will have a 'backpropagate' method to calculate errors.
    - It will have a 'train' method to update the weights.
    """

    def __init__(self, input_size, hidden_size, output_size):
        # Constructor method that is called automatically and initializes the network's weights and biases

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # np.random.rand(rows, cols) makes a matrix filled with numbers between 0.0 and 1.0
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size) # weights between input layer and hidden layer (2 rows 2 collumns)
        self.weights_output_hidden = np.random.rand(self.hidden_size, self.output_size) # weights between hidden layer and output layer ( 2 rows 1 collumn)

        # We initialize the biases in a single row rather than a whole matrix
        self.bias_hidden = np.random.rand(1, self.hidden_size) # Biases for hidden layer
        self.bias_output  = np.random.rand(1, self.output_size) # Biases for output layer

        print("Neural Network initialized with:")
        print(f"  Input Layer Size: {self.input_size}")
        print(f"  Hidden Layer Size: {self.hidden_size}")
        print(f"  Output Layer Size: {self.output_size}")



    def feedforward(self, inputs):
        '''
        This is our forward propagation or forward pass method.
        It takes our inputs and applies weights and adds biases to find the prediction which is then passed through the activation function.
        '''

        # np.dot() performs matrix multiplication with the two matrixes inputs and weights_input_hidden
        # Inputs would be [0,1] which is a 2d array with 1 row and 2 collumns
        # weights_input_hidden has 2 rows and 2 collumns

        """
        [[ w_i1_h1,  w_i1_h2 ],
        [ w_i2_h1,  w_i2_h2 ]]
        """

        hidden_raw = np.dot(inputs, self.weights_input_hidden)
        hidden_biased = hidden_raw + self.bias_hidden # We add the bias to the dot product
        self.hidden_layer_output = sigmoid(hidden_biased) # Finally, we pass it all through the activation function to get an output

        # Repeat for output node to get final output from the forward pass
        output_raw = np.dot(self.hidden_layer_output, self.weights_output_hidden)
        output_biased = output_raw + self.bias_output
        output = sigmoid(output_biased)

        # We store this as self so we can access at any time
        self.output = output
        return self.output

    def train(self, X, y, epochs, rate):
        """
        This is our training loop. 
        X is our training input data (XOR pairs)
        y is the target output data ([0, 1, 1, 0])
        epochs are the amount of times we loop/train the dataset
        The rate is a small number we use to scale our weights. Higher value means quicker but less efficient trainng. Lower value means more efficient but slower training.
        """

        print(f"\n--- Starting Training ---")
        print(f"Epochs: {epochs}, Learning Rate: {rate}")

        for i in range(epochs):
            self.feedforward(X) # We do a forward pass with all of our pairs to get predicted outputs

            error = y - self.output # Y is the correct answer from the XOR table, how far off was our output?

            if(i + 1) % 1000 == 0:
                loss = np.mean(np.abs(error)) # Mean absolute error
                print(f"Epoch {i + 1}/{epochs}, Loss: {loss:.6f}")
        
        # It asks, "How wrong were we, and how 'steep' was the curve at our guess?"
        delta = error * sigmoid_der(self.output)

        # We multiply the error by the weights that connect them
        # .T flips the matrix as we are propagating backwards
        # # (delta: 4x1) * (weights_ho.T: 1x2) -> (hidden_error: 4x2)
        hidden_error = np.dot(delta, self.weights_output_hidden.T)
        hidden_delta = hidden_error * sigmoid_der(self.hidden_layer_output)

        # Gradient Descent: We descend by adjusting the weights in the direction that reduces the error
        
        # Updates weights and biases
        self.weights_output_hidden += np.dot(self.hidden_layer_output.T, delta) * rate # # (hidden_output.T: 2x4) * (output_delta: 4x1) -> (adjustment: 2x1)
        self.bias_output += np.sum(delta, axis = 0, keepdims=True) * rate # Biases are simpler, we just "sum" the deltas

        # Do the same for the input layer (output layer first because we are moving backwards)
        # (X.T: 2x4) . (hidden_delta: 4x2) -> (adjustment: 2x2)
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * rate

        print("Training complete!")



if __name__ == "__main__":
    # This block of code only runs if you execute this .py file directly

    # Our 4 inputs from our XOR dataset
    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])

    # Our corresponding TARGET data for X
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])


    
    print("--- Testing our Neural Network Class ---")
    
    # Create our XOR network: 2 inputs, 2 hidden neurons, 1 output
    nn = neuralNetwork(input_size=2, hidden_size=2, output_size=1)
    
    print("\nInitial Weights (Input -> Hidden):\n", nn.weights_input_hidden)
    print("\nInitial Bias (Hidden):\n", nn.bias_hidden)
    print("\nInitial Weights (Hidden -> Output):\n", nn.weights_output_hidden)
    print("\nInitial Bias (Output):\n", nn.bias_output)

    # (We'll use 4 hidden neurons for better results with XOR)
    nn = neuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    # We pass in our data, # of epochs, and a learning rate
    nn.train(X, y, epochs=10000, rate=0.1)
    
    # --- 4. Test the Network after Training ---
    print("\n--- Testing the Network After Training ---")
    
    # Make a prediction for each XOR case
    for inputs in X:
        # Reshape the 1D input array [0, 0] to a 2D array [[0, 0]]
        # so it's a "batch" of 1, matching the format our
        # feedforward method expects.
        test_input = inputs.reshape(1, 2)
        
        prediction = nn.feedforward(test_input)
        
        # We round the prediction to 0 or 1
        # Access the single number at [0][0]
        rounded_prediction = 1 if prediction[0][0] > 0.5 else 0
        
        # Access the single number at [0][0] for printing
        print(f"Input: {inputs}, Prediction: {prediction[0][0]:.4f}, Rounded: {rounded_prediction}")





