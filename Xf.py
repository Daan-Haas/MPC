import numpy as np
import control as ct
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

P, L, K = ct.dare(A, B, np.identity(2), R=[1])
x2to_try = np.linspace(-0.9, 0.9, 100)
x1to_try = np.linspace(-0.4, 0.4, 100)

statestotry = [[x1to_try[i], x2to_try[j]] for j in range(len(x1to_try)) for i in range(len(x1to_try))]
Xn = [(np.matmul(np.array(state), np.matmul(P, np.array(state).T))) < 0.85 for state in statestotry]


# Separate the coordinates based on the boolean values
true_points = [statestotry[i] for i in range(len(statestotry)) if Xn[i]]
false_points = [statestotry[i] for i in range(len(statestotry)) if not Xn[i]]

# Plot the points
true_points = np.array(true_points)
false_points = np.array(false_points)
plt.scatter(false_points[:, 0], false_points[:, 1], color='red', label='X', marker=',')
plt.scatter(true_points[:, 0], true_points[:, 1], color='green', label='Xf', marker=',')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('set of controllable states')
plt.legend()
plt.grid(True)
plt.show()


Xn = [np.matmul(-K, state) < np.deg2rad(20) for state in true_points]
print(sum(Xn))
print(len(true_points))
print([true_points[i] for i in range(len(true_points)) if not Xn[i]])

