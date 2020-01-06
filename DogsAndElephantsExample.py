'''
Dog vs. Elephant Classifier
Last Revision: October 20, 2019
'''

# Import numpy library as assign to "np"
import numpy as np

# Import matplotlib for training graph
from matplotlib import pyplot as plt

# Reading data from text file
fo = open("data.txt", "r")
dataFile = fo.readlines()
fo.close()

data = []
for line in dataFile:
    line = line.strip()
    data.append(line.split(','))

for row in range(len(data)):
    for point in range(len(data[row])):
        data[row][point] = int(data[row][point])

print(f"Data imported from text file:\n{data}")

### --- MAIN FLOW STARTS --- ###

# Define the sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Define the sigmoid derrivative function
def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Weights and bias
w1 = np.random.randn()
w2 = np.random.randn()
b = 0.01

# NETWORK PARAMETERS
LEARNING_RATE = 0.001
EPOCHS = 500000
REDUCTION_FACTOR = 500

# Optional - collect cost samples for analysis
cost_memory = []
interval = 500

# MAIN LOOP
for count in range(EPOCHS):
    
    # Choose a random data point
    point = data[np.random.randint(len(data))]

    # FORWARD PROPOGATION - network makes a prediction
    z = ((point[0] * w1) + (point[1] * w2) + b)/REDUCTION_FACTOR
    prediction = sigmoid(z)

    # CALCULATE COST - see how far off the prediciton was
    target = point[2]
    cost = np.square(prediction - target)

    # Save cost to memory if applicable
    if (count%interval == 0):
        cost_memory.append(cost)

    #  BACKPROPOGATION BEGINS - make small adjustments to improve network
    # Find the derrivative of cost with respect to the prediction
    dcost_dprediction = 2 * (prediction - target)

    # Find the derrivative of the prediction with respect to the raw output (sum of weights*inputs + bias)
    dprediction_dz = sigmoid_d(z)

    # Find the derrivative of the raw output with respect to each of w1, w2, and b
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    # Combine the above derrivatives to find the derrivative of cost with respect to each of w1, w2, and b
    dcost_dw1 = dcost_dprediction * dprediction_dz * dz_dw1
    dcost_dw2 = dcost_dprediction * dprediction_dz * dz_dw2
    dcost_b = dcost_dprediction * dprediction_dz * dz_db

    # Using the calculated partial derrivates, update the weights and bias
    w1 = w1 - (LEARNING_RATE * dcost_dw1)
    w2 = w2 - (LEARNING_RATE * dcost_dw2)
    b = b - (LEARNING_RATE * dcost_b)

print(f"Training complete! Final network values:\nW1: {w1}\nW2: {w2}\nB: {b}")

# Plot cost over time
print(cost_memory)
plt.plot(cost_memory)
plt.show()
