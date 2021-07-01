import numpy as np
import matplotlib.pyplot as plt


# Create a 5x5 identity matrix :
def identity_matrix(n):
    identity = np.array(n)
    return identity


# Import data from data.txt and parse it in tow lists :
def parse_data(file_name, x, y):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            x.append(float(line.split(',')[0]))
            y.append(float(line.split(',')[1]))
    return x, y


# Plotting Data using matplotlib package :
def plot_data(x, y):
    # display the plot
    plt.scatter(x, y, marker='x')
    # x-axis label title :
    plt.xlabel('Profit in $10,000s')
    # y-axis label title :
    plt.ylabel('Population of City in 10,000s')

