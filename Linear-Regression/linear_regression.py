# import numpy package
import numpy as np
import matplotlib.pyplot as plt
print('[INFO]: All Package are imported Successfully ...')

def identity(n):
    id = np.identity(n)
    return id

id = identity(5)
print('[INFO]: Creating 5x5 Identity Matrix ...')
print(id)

# reading .txt file and return data in a list 
def read_file(filename):
    with open(filename) as file:
        data = file.readlines()
    return data

# parsing data into two list population and profit
def parse_data(L):
    x = []
    y = []
    
    for i in L:
        x.append(float(i.split(',')[0]))
        y.append(float(i.split(',')[1]))
    
    
    return np.array(x).reshape(len(x),1),np.array(y).reshape(len(x),1)

# File Path
file = 'Linear-Regression/food_profit_data.txt'

# initalize needed lists :
population = []
profit = []

# Load and parse data:
data = read_file(file)
print('[INFO]: File is Readed Successfully ...')
population, profit = parse_data(data)
print('[INFO]: File is parsed Successfully ...')
print('[INFO]: Parsed Data :\n\nPopulation = ', population, '\n\nProfit = ', profit, '\n')
print('[INFO]: Each Array Contain ', len(population), 'elements ...')

# define plot_data function :
def plot_data(x, y):
    print('[INFO]: Plotting Data ...')
    plt.scatter(x, y, marker='x')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.title('Figure 1: Scatter plot of training data')
    plt.show()
    
plot_data(population, profit)

# add a new column of ones to population List
X_population = np.append(np.ones((len(population), 1)), population, axis=1)

# initialize fitting parameters
theta = np.zeros((2,1))

iterations = 1500
alpha = 0.01

print('[INFO]: All needed variables are initilized successfully...')

def compute_cost(x, y, theta):
    # number of training exemples
    m = len(y)    
    # initialize cost function result
    J = 0
    
    # Compute the cost of particular choice of theta
    predictions = x.dot(theta)
    sqerrors = (predictions - y) ** 2
    
    J = 1/(2*m) * np.sum(sqerrors)
    
    return J, m

J, m = compute_cost(X_population, profit, theta)
print(f'[INFO]: we have in this example {m} training examples ...')
print(f'[INFO]: With theta = {theta}, Cost Computed {J}...')
print(f'[INFO]: Expected cost value (approx) = {np.round(J, 2)}\n')
J = compute_cost(X_population, profit, [[-1] , [2]])

def gradiant_descent(x, y, theta, alpha, iterations):
    # number of training exemples
    m = len(y)
    J_history = np.zeros((iterations, 1))
    
    for iter in range(iterations):
        predictions = x.dot(theta)
        updates = np.transpose(x).dot(predictions - y)
        theta = (theta - alpha * (1/m) * updates)
        
    print(f'[INFO]: theta found by gradiant descent : \n{theta}')
    print(f'[INFO]: theta found by gradiant descent (approx): \n{np.round(theta, 4)}')
    
    return theta

theta_n = gradiant_descent(X_population, profit, theta, alpha, iterations)

def plot_trained_data(x, y, z):
    plt.scatter(y, z, marker='x', label='Training data')
    plt.plot(y, x.dot(theta_n), 'r', label='Linear Regression')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.title('Figure 1: Scatter plot of training data')
    plt.legend()
    plt.show()

plot_trained_data(X_population,X_population[:,1], profit)

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta_n)
print(f'[INFO]: For population = 35.000 we predict a profit of {np.round(predict1*10000, 4)}')

predict2 = np.array([1, 7]).dot(theta_n)
print(f'[INFO]: For population = 35.000 we predict a profit of {np.round(predict2*10000, 4)}')

# Grid over which will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))


# fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i][j], m = compute_cost(X_population, profit, t)


J_vals = np.transpose(J_vals)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals,cmap='viridis', edgecolor='none')
plt.show()

fig, ax = plt.subplots()
cmap = ax.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
fig.colorbar(cmap)
plt.show()

