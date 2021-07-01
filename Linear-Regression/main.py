import numpy as np
import matplotlib.pyplot as plt

# Variables
X = []
Y = []
m = 0
iteration = 
file_name = 'Linear-Regression/data.txt'


# Create a 5x5 identity matrix :
def identity_matrix(n):
    identity = np.identity(n)
    return identity


# Import data from data.txt and parse it in tow lists :
def parse_data(file, x, y):
    f = open(file, 'r')
    # read lines and parse data :
    print('[INFO]: Parsing Data ...\n')
    for line in f.readlines():
        x.append(float(line.split(',')[0]))
        y.append(float(line.split(',')[1]))
    print('X = ', X)
    print('Y = ', Y)
    m = len(X)
    print('\n[INFO]: We have ', len(X), ' Training Examples\n')
    return x, y, m


# Plotting Data using matplotlib package :
def plot_data(x, y):
    print('[INFO]: Plotting Data ...')
    # display the plot
    plt.scatter(x, y, marker='x')
    # x-axis label title :
    plt.xlabel('Profit in $10,000s')
    # y-axis label title :
    plt.ylabel('Population of City in 10,000s')
    print('[INFO]: Plot is ready ...')
    plt.show()
    # Save Plot into graphs :
    plt.savefig('Linear-Regression/graphs/graph_1.png')
    print('[INFO]: Picture is save in graphs ...')


if __name__ == "__main__":
    # execute only if run as a script
    Id = identity_matrix(5)
    print('[INFO]: 5x5 Identity Matrix :\n', Id, '\n')
    (X, Y, m) = parse_data(file_name, X, Y)
    plot_data(X, Y)


