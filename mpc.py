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

Q = np.array([[1, 0], [0, 0.0001]])
P, L, K = ct.dare(A, B, Q, R=[1])

model.set_expression(expr_name='terminal cost', expr=0.5 * np.matmul(xvec.T, np.matmul(0.1*P, xvec)).squeeze())
model.set_expression(expr_name='stage cost', expr=0.5 * (np.matmul(xvec.T, np.matmul(Q, xvec))).squeeze())

model.setup()
mpc = do_mpc.controller.MPC(model)
setup_mpc = dict(n_horizon=10, t_step=dt, state_discretization='collocation', store_full_solution=True)

mpc.set_param(**setup_mpc)
lterm = model.aux['stage cost']
mterm = model.aux['terminal cost']

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(delta=1)  # input penalty

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

rhs = []
lhs = []

for k in range(500):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    rhs.append(0.5 * np.matmul(x0.T, np.matmul(P, x0)).squeeze() -
              (0.5 * np.matmul(x0.T, np.matmul(Q, x0)).squeeze() + abs(u0.squeeze())))
    x0 = estimator.make_step(y_next)
    lhs.append(0.5 * np.matmul(x0.T, np.matmul(10 * P, x0)).squeeze())

rcParams['axes.grid'] = True
rcParams['font.size'] = 18
fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(20, 9))
graphics.plot_results()
fig.suptitle(f'MPC controller', y=1)
# graphics.reset_axes()
plt.show()

plt.plot(rhs)
plt.plot(lhs)
plt.legend(['$V_f(x) - l(x, u)$', '$V_f(f(x, u))$'])
plt.show()
