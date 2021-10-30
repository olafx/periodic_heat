import numpy as np
import matplotlib.pyplot as plt

t_a = .0
t_b = 10.0
n = 2001
P = 1.0
p = 2.0

t = np.linspace(t_a, t_b, n)
f = np.linspace(-0.5*(n-1)/(t_b-t_a), 0.5*(n-1)/(t_b-t_a), n)

y = P/2*(1+np.exp(2j*np.pi*t/p))

Y = np.fft.fftshift(np.fft.fft(y)/n)

Y2 = np.zeros(n, dtype='complex')
Y2[n//2] = P/2
Y2[n//2+round((t_b-t_a)/p)] = P/2

y2 = np.fft.ifft(np.fft.fftshift(Y2)*n)

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# ax3 = fig.add_subplot(413)
# ax4 = fig.add_subplot(414)

# ax1.plot(t, np.real(y), c='red', label='real')
# ax1.plot(t, np.imag(y), c='blue', label='imaginary')
# ax1.set_xlabel('$t$ [s]', size=14)
# ax1.set_ylabel('$P(t)$ [a.u.]', size=14)

# ax2.plot(f, np.real(Y), c='red', label='real')
# ax2.plot(f, np.imag(Y), c='blue', label='imaginary')
# ax2.set_xlim([-2,2])
# ax2.set_ylim([0, 0.01])
# ax2.set_title('$n$ '+str(n))
# ax2.set_xlabel('$f$', size=14)
# ax2.set_ylabel('$\mathcal{F}_t\{P(t)\}(f)$', size=14)

# ax3.plot(f, np.real(Y2), c='red')
# ax3.set_xlim([-5,5])

# ax4.plot(t, np.real(y2-y), c='red', label='real')
# ax4.plot(t, np.imag(y2-y), c='blue', label='imaginary')
# ax4.set_xlabel('$t$ [s]')
# ax4.set_ylabel('iFFT - signal [a.u.]')

ax1.plot(f, np.real(Y), c='red', alpha=0.5)
ax1.plot(f, np.imag(Y), c='blue', alpha=0.5)
ax1.plot(f, np.real(Y), '.', c='red', label='real')
ax1.plot(f, np.imag(Y), '.', c='blue', label='imaginary')
ax1.set_xlim([-1,1.5])
# ax1.set_ylim([0, 0.01])
ax1.set_title('$n$ '+str(n))
ax1.set_xlabel('$f$ [1/s]', size=14)
ax1.set_ylabel('$FFT$ [a.u.]', size=14)
plt.legend()

ax2.plot(f, np.real(Y), c='red', alpha=0.5)
ax2.plot(f, np.imag(Y), c='blue', alpha=0.5)
ax2.plot(f, np.real(Y), '.', c='red', label='real')
ax2.plot(f, np.imag(Y), '.', c='blue', label='imaginary')
ax2.set_xlim([-1,1.5])
ax2.set_ylim([0, 0.005])
ax2.set_title('$n$ '+str(n))
ax2.set_xlabel('$f$ [1/s]', size=14)
ax2.set_ylabel('$FFT$ [a.u.]', size=14)
plt.legend()

plt.tight_layout()

plt.savefig('FFT sine 5.png', dpi=200, bbox_inches='tight')
