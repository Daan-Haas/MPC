import numpy as np
import control as ct

import matplotlib.pyplot as plt

I = 3.28
m = 87
g = 9.81
h = 1
a = 0.5
b = 1.0
v_r = 5
dt = 0.001
T = 20

x_0 = np.array([[np.deg2rad(5)], [0]])
A = np.array([[0, 1], [m*g*h/(I + m*h**2), 0]])
B = np.array([[(-m*h*a*v_r)/(b*(I + m*h**2))],
              [(-m*h*v_r*2)/(b*(I + m*h**2))]])
C = np.array([1, 0])
D = np.array([0])

K = ct.acker(A, B, (-3.01 + 3.01j, -3.01 - 3.01j))
x_next = x_0
u = np.array([0])


X = [x_next[0][0]]
U = [u[0]]

for i in range(1000):
    u = np.matmul(K, x_next)
    dx = np.matmul(A - np.matmul(K, B), x_next)
    x_next += dx * dt
    X.append(x_next[0][0])
    U.append(u[0][0])

fig, ax = plt.subplots(2)
ax[0].plot(np.linspace(0, stop=len(X)*dt, num=len(X)), np.rad2deg(X))
ax[0].set(ylabel='Roll angle [deg]')
ax[1].plot(np.linspace(0, stop=len(X)*dt, num=len(X)), np.rad2deg(U))
ax[1].set(ylabel='Steer angle [deg]', xlabel='Time [s]')
fig.suptitle('Closed loop controller')
plt.show()
