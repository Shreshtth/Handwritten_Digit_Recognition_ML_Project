# Handwritten_Digit_Recognition_ML_Project
I made a machine learning project that can detect any digit (0-9), using a 28*28 pixel image. 

# Data Set used-
MNIST dataset present is Keras Lib, consisting almost 60000 training exmaples and 10000 testing examples.

# Model Explanation:
1. A neural network consisting of three 'Dense'(Fully Connected) layers, the first two layers having 128 units each and last layer having 10 units. 
2. First two layers utilizes the 'Relu' activation whereas last layer is using 'Softmax' activation, which gives the final output.
3. While, compiling the model, 'Adam Optimizer' is used with a learning rate= 0.001.
