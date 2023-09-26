import numpy as np

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

n = 20 
x = range(n)
y_input = np.ones(n) * 1e10

y_meas = y_input + np.random.normal(loc=0, scale=1e9, size=(n))
y_err = np.ones(n) * 1e9


dy = y_meas - y_input
dy_upper = (y_meas + y_err) - y_input
dy_lower = (y_meas - y_err) - y_input

log_dy = np.log10(y_meas) - np.log10(y_input)
log_dy_upper = np.log10((y_meas + y_err)) - np.log10(y_input)
log_dy_lower = np.log10((y_meas - y_err)) - np.log10(y_input)

log_yerr_upper = log_dy_upper - log_dy
log_yerr_lower = log_dy - log_dy_lower

# plot 
fig, ax = plt.subplots()
ax.plot(x, dy)
ax.fill_between(x, dy_lower, dy_upper, alpha=0.5)

# plot 
fig, ax = plt.subplots()
ax.plot(x, log_dy)
ax.fill_between(x, log_dy_lower, log_dy_upper, alpha=0.5)


# plot 
fig, ax = plt.subplots()
ax.plot(x, y_input)
ax.errorbar(x=x, y=y_meas, yerr=y_err)
ax.set_yscale("log")

fig, ax = plt.subplots()
ax.plot(x, y_input)
ax.errorbar(x=x, y=np.log10(y_meas) - np.log10(y_input))



ax.set_yscale("log")