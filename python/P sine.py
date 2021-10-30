import numpy as np
import matplotlib.pyplot as plt

P = 1.0
p = 10.0
t_i = 0.0
t_f = 60.0
n = 1001

t = np.linspace(t_i, t_f, n)
f = np.linspace(-0.5*(n-1)/(t_f-t_i), 0.5*(n-1)/(t_f-t_i), n)

y = P/2*(1+np.exp(2j*np.pi*t/p))
Y = np.fft.fftshift(np.fft.fft(y)/n)

for i in range(0, n):
    Y[i] -= 1j*np.imag(Y[i])

fig = plt.figure(figsize=(6, 7))

ax1 = fig.add_subplot(211)
ax1.plot(t, np.real(y), c='red', label='real')
ax1.plot(t, np.imag(y), c='blue', label='imaginary')
ax1.set_xlabel('$t$ [s]', size=14)
ax1.set_ylabel('$P(t)$ [W]', size=14)

ax2 = fig.add_subplot(212)
ax2.plot(f, np.real(Y), '.', c='red', label='real')
ax2.plot(f, np.imag(Y), '.', c='blue', label='imaginary')
ax2.plot(f, np.real(Y), c='red', alpha=0.5)
ax2.plot(f, np.imag(Y), c='blue', alpha=0.5)
ax2.set_xlabel('$f$ [1/s]', size=14)
ax2.set_ylabel('$\mathcal{F}_t\{P(t)\}(f)$ [W]', size=14)
ax2.set_xlim([-0.1, 0.2])

plt.tight_layout()
plt.legend()

plt.savefig('P sine.png', dpi=200, bbox_inches='tight')
