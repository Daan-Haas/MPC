import numpy as np
import do_mpc
import scipy

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

model_type = 'discrete'
model = do_mpc.model.Model(model_type)

x = model.set_variable(var_type='_x', var_name='x', shape=(2, 1))
delta = model.set_variable('_u', 'delta', shape=1)

x_next = A@x + B@delta

model.set_rhs('x', x_next)

P = scipy.linalg.solve_discrete_are(A, B, np.identity(2), np.array([1]))
xvec = np.array([[x[0]], [x[0]]])
Q = np.identity(2)
R = np.identity(1)
model.set_expression(expr_name='terminal cost', expr=0.5 * np.matmul(xvec.T, np.matmul(P, xvec)).squeeze())
model.set_expression(expr_name='stage cost', expr=0.5 * (np.matmul(xvec.T, np.matmul(Q, xvec)) + delta**2))

model.setup()

mpc = do_mpc.controller.MPC(model)
setup_mpc = dict(n_horizon=5, t_step=dt, state_discretization='discrete', store_full_solution=True)

mpc.set_param(**setup_mpc)
lterm = model.aux['stage cost']
mterm = model.aux['terminal cost']

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(delta=10)  # input penalty


mpc.terminal_bounds['lower', 'x'] = np.array([[np.deg2rad(-35)], [np.deg2rad(-35)]])
mpc.terminal_bounds['upper', 'x'] = np.array([[np.deg2rad(35)], [np.deg2rad(35)]])
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
e = np.ones([model.n_x, 1])
x0 = np.random.uniform(np.deg2rad(-11)*e, np.deg2rad(11)*e)  # Values between +3 and +3 for all states
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0
print(x0)
# Use initial state to set the initial guess.
mpc.set_initial_guess()

# %%capture
for k in range(100):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

rcParams['axes.grid'] = True
rcParams['font.size'] = 18

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(16, 9))
graphics.plot_results()
graphics.reset_axes()
plt.show()

