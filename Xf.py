import numpy as np
import scipy
import matplotlib.pyplot as plt

# Constants
I = 3.28
m = 87
g = 9.81
h = 1
a = 0.5
b = 1.0
v_r = 5
dt = 0.001
T = 20

# System dynamics
A = np.array([[0, 1], [m*g*h/(I + m*h**2), 0]])
B = np.array([[(-m*h*a*v_r)/(b*(I + m*h**2))],
              [(-m*h*v_r*2)/(b*(I + m*h**2))]])

# Calculate the optimal control
P = scipy.linalg.solve_discrete_are(A, B, np.identity(2), np.array([1]))

x2to_try = np.linspace(-1.2, 1.2, 100)
x1to_try = np.linspace(-0.5, 0.5, 100)

statestotry = [[x1to_try[i], x2to_try[j]] for j in range(len(x1to_try)) for i in range(len(x1to_try))]
Xn = [(np.matmul(np.array(state), np.matmul(P, np.array(state).T))) < 1.7 for state in statestotry]


# Separate the coordinates based on the boolean values
true_points = [statestotry[i] for i in range(len(statestotry)) if Xn[i]]
false_points = [statestotry[i] for i in range(len(statestotry)) if not Xn[i]]

# Plot the points
true_points = np.array(true_points)
false_points = np.array(false_points)
plt.scatter(false_points[:, 0], false_points[:, 1], color='red', label='X', marker=',')
plt.scatter(true_points[:, 0], true_points[:, 1], color='green', label='Xn', marker=',')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('set of controllable states')
plt.legend()
plt.grid(True)
plt.show()

