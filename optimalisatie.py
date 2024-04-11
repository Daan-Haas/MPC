import numpy as np
import do_mpc
import control as ct

from matplotlib import rcParams
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

A = np.array([[0, 1], [m*g*h/(I + m*h**2), 0]])
B = np.array([[(-m*h*a*v_r)/(b*(I + m*h**2))],
              [(-m*h*v_r*2)/(b*(I + m*h**2))]])
C = np.array([1, 0])
D = np.array([0])

model_type = 'continuous'
model = do_mpc.model.Model(model_type)

theta = model.set_variable(var_type='_x', var_name='theta')
x2 = model.set_variable(var_type='_x', var_name='x2')
delta = model.set_variable('_u', 'delta', shape=1)
xvec = np.array([[theta], [x2]])

x_next = A@xvec + B@delta

model.set_rhs('theta', x_next[0])
model.set_rhs('x2', x_next[1])


P, L, K = ct.dare(A, B, np.eye(2), R=[1])
Q = 10000 * np.array([[1, 0], [0, 0.0001]])

model.set_expression(expr_name='terminal cost', expr=0.5 * np.matmul(xvec.T, np.matmul(10 * P, xvec)).squeeze())
model.set_expression(expr_name='stage cost', expr=0.5 * (np.matmul(xvec.T, np.matmul(Q, xvec))).squeeze())

model.setup()

us = []
x1s = []
x2s = []

R_to_try = [0, 1, 10, 1000]

for i in R_to_try:

    mpc = do_mpc.controller.MPC(model)
    setup_mpc = dict(n_horizon=10, t_step=dt, state_discretization='collocation', store_full_solution=True)

    mpc.set_param(**setup_mpc)
    lterm = model.aux['stage cost']
    mterm = model.aux['terminal cost']

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(delta=i)  # input penalty

    mpc.terminal_bounds['lower', 'theta'] = np.deg2rad(-35)
    mpc.terminal_bounds['upper', 'theta'] = np.deg2rad(35)
    mpc.bounds['lower', '_u', 'delta'] = np.deg2rad(-20)
    mpc.bounds['upper', '_u', 'delta'] = np.deg2rad(20)

    mpc.setup()
    estimator = do_mpc.estimator.StateFeedback(model)
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=dt)
    simulator.setup()

    # Seed
    np.random.seed(99)

    # Initial state
    x0 = np.array([[np.deg2rad(5)], [0]])
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    u = []
    x1 = [x0[0]]
    x2array = [x0[1]]
    for k in range(500):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
        u.append(u0.squeeze())
        x1.append(x0[0])
        x2array.append(x0[1])

    us.append(u)
    x1s.append(x1)
    x2s.append(x2array)

    rcParams['axes.grid'] = True
    rcParams['font.size'] = 18
    fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(20, 9))
    graphics.plot_results()
    fig.suptitle(fr'stage cost R equal to {i}', y=1)
    # graphics.reset_axes()
plt.show()

# feedback controller
uFB = [0]
x1FB = [np.deg2rad(5)]
x2FB = [0]

Kpp = ct.acker(A, B, (-3.01 + 3.01j, -3.01 - 3.01j))
x_next = np.array([[np.deg2rad(5)], [0]])
for i in range(500):
    u = np.matmul(Kpp, x_next)
    dx = np.matmul(A - np.matmul(Kpp, B), x_next)
    x_next += dx * dt
    x1FB.append(x_next[0][0])
    x2FB.append(x_next[1][0])
    uFB.append(u.squeeze())


# Plotting response for different horizons and FB
us.append(uFB)
for u in us:
    plt.plot(np.linspace(0, len(u)*dt, len(u)), np.rad2deg(u))
plt.title('Steering inputs for various stage cost component R')
plt.xlabel('time [s]')
plt.ylabel('steer angle [deg]')
plt.legend(['0', '1', '10', '1000', 'Feedback'], title='R')
plt.show()

x1s.append(x1FB)
for x in x1s:
    plt.plot(np.linspace(0, len(x)*dt, len(x)), np.rad2deg(x))
plt.title('roll angle for various stage cost component R')
plt.xlabel('time [s]')
plt.ylabel('roll angle [deg]')
plt.legend(['0', '1', '10', '1000', 'Feedback'], title='R')
plt.show()

x2s.append(x2FB)
for x in x2s:
    plt.plot(np.linspace(0, len(x)*dt, len(x)), np.rad2deg(x))
plt.title('steer rate for various stage cost component R')
plt.xlabel('time [s]')
plt.ylabel('steer rate [deg/s]')
plt.legend(['0', '1', '10', '1000', 'Feedback'], title='R')
plt.show()
