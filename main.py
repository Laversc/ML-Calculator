from data import get_mnist

import numpy as np
import matplotlib.pyplot as plt

#Defining the activation function used (Sigmoid)
def sig(x):
    return (1 / (1 + np.exp(-x)))


images, labels = get_mnist()
weights_in_h = np.random.uniform(-0.5, 0.5, (20, 784))
weights_h_out = np.random.uniform(-0.5, 0.5, (10, 20))
bias_in_h = np.zeros((20, 1))
bias_h_out = np.zeros((10, 1))


learn_rate = 0.01
nr_correct = 0
epochs = 4
acc = 0


for epoch in range(epochs):
    for img, label in zip(images, labels):
        img.shape += (1,)
        label.shape += (1,)

        # Forward propagation input -> hidden

        #Matrix multiplication of the input values with the weights_in_h matrix and then add the bias_in_h matrix
        hidden = weights_in_h @ img + bias_in_h

        #Using the sigmoid activation function to get the values of the hidden layer neuromns
        hidden = sig(hidden)

        # Forward propagation hidden -> output
        out = weights_h_out @ hidden + bias_h_out
        out = sig(out)

        # Cost / Error calculation

        #Mean squared error function
        mean_sqared_err = 1 / len(out) * np.sum((out - label) ** 2, axis=0)

        #Determine if the classification was accurate - if so increase acc counter 
        acc += int(np.argmax(out) == np.argmax(label))

        # Backpropagation output -> hidden (cost function derivative)

        #error of the output layer to the label
        out_err = out - label

        #Calculating the update values for the hidden layer
        weights_h_out -= learn_rate * out_err @ np.transpose(hidden)
        bias_h_out -= learn_rate * out_err

        # Backpropagation hidden -> input (activation function derivative)
        in_err = np.transpose(weights_h_out) @ out_err * (hidden * (1 - hidden))
        weights_in_h -= learn_rate * in_err @ np.transpose(img)
        bias_in_h -= learn_rate * in_err

    # Show accuracy for this epoch
    print(f"Accuracy : {round((acc / images.shape[0]) * 100, 2)}%")

    acc = 0

# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()