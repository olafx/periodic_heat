import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(
    np.linspace(-1, 1, 1000),
    np.linspace(-1, 1, 1000))
k_x, k_y = np.meshgrid(
    np.linspace(-0.5*(1000-1)/2.0, 0.5*(1000-1)/2.0, 1000),
    np.linspace(-0.5*(1000-1)/2.0, 0.5*(1000-1)/2.0, 1000))

Q = 1.0/(2.0*np.pi*0.05*0.03)*np.exp(-0.5*((x/0.05)**2+(y/0.03)**2))
F_Q = np.fft.fftshift(np.fft.fftn(Q)/1000/1000)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
img1 = ax1.pcolormesh(x, y, Q, shading='auto')
cb1 = fig.colorbar(img1)
cb1.set_label('$Q$', size=14)
ax1.set_xlabel('$x$ [cm]', size=14)
ax1.set_ylabel('$y$ [cm]', size=14)
ax1.set_xlim([-0.2,0.2])
ax1.set_ylim([-0.2,0.2])
ax2 = fig.add_subplot(122)
img2 = ax2.pcolormesh(k_x, k_y, np.abs(F_Q), shading='auto')
cb2 = fig.colorbar(img2)
cb2.set_label('$\mathcal{F}_{xy}\{Q\}$', size=14)
ax2.set_xlabel('$k_x$ [1/cm]', size=14)
ax2.set_ylabel('$k_y$ [1/cm]', size=14)
ax2.set_xlim([-15,15])
ax2.set_ylim([-15,15])

plt.tight_layout()

# plt.show()
plt.savefig('Q.png', dpi=200, bbox_inches='tight')
