import numpy as np
import matplotlib.pyplot as plt

P = 1.0
p = 10.0
t_i = 0.0
t_f = 60.0
n = 2000
m = 10

t = np.linspace(t_i, t_f, n)
f = np.linspace(-0.5*(n-1)/(t_f-t_i), 0.5*(n-1)/(t_f-t_i), n)

Y = np.empty(n, dtype='complex')
k = -1
for i in range(n//2-1, -1, -2):
    if f[i-1] < k/p and f[i+1] > k/p:
        Y[i+1] = (1j*P)/(k*np.pi)
        k -= 2
    if k == -1-2*m:
        break
Y[n//2] = P/2
k = 1
for i in range(n//2+1, n-1):
    if f[i-1] < k/p and f[i+1] > k/p:
        Y[i+1] = (1j*P)/(k*np.pi)
        k += 2
    if k == 1+2*m:
        break

y = np.fft.ifft(Y*n)

fig = plt.figure(figsize=(8, 9))

ax1 = fig.add_subplot(312)
ax1.plot(t, np.abs(y), c='black')
ax1.set_xlabel('$t$ [s]', size=14)
ax1.set_ylabel('iFFT [a.u.]', size=14)
ax1.set_xlim([0, 60])

ax3 = fig.add_subplot(313)
ax3.plot(t, np.abs(y), c='black')
ax3.set_xlabel('$t$ [s]', size=14)
ax3.set_xlim([3, 6])
ax3.set_ylim([-0.05, 1.15])
ax3.plot([5.0, 5.0], [-1.0, 2.0], '--', c='black')
ax3.plot([2.0, 7.0], [0.0, 0.0], '--', c='black')
ax3.set_xlabel('$t$ [s]', size=14)
ax3.set_ylabel('iFFT [a.u.]', size=14)

ax2 = fig.add_subplot(311)
ax2.plot(f, np.real(Y), '.', c='red', label='real')
ax2.plot(f, np.imag(Y), '.', c='blue', label='imaginary')
ax2.set_xlabel('$f$ [1/s]', size=14)
ax2.set_xlim([-5, 5])
ax2.set_ylabel('Finite # of Frequencies $f$ [a.u.]', size=14)

plt.tight_layout()
plt.legend()

# plt.show()
plt.savefig('gibbs.png', dpi=200, bbox_inches='tight')
