# Handwritten Image Recognition Based on Convolutional Neural Networks in Raspberry Pi

This project uses convolutional neural network to train a model using the MNIST dataset and Tensorflow framework which can be used to recognise handwritten numbers.
The image can be captured through Pi camera and it directly accepts the image stream without saving file locally which helps in improving the latency.
NeuralConvo.py file is used to train the model using MNIST Dataset and NeuralConvo_test-pi.py file is used to implement trained model on Raspberry Pi.
bias.txt files contain the value of biases generated and test.txt files contain the weights generated while training of model.
