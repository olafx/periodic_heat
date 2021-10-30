import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0, 1, 1000)
y = 0.5*np.exp(2*z)/(1-np.exp(2*2))-np.exp(-2*z)/(1-np.exp(-2*2))

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)

ax.plot(z, y, c='black')
ax.set_xlim([0, 1])
ax.set_xlabel('Ratio $z/d$ [a.u.]', size=14)
ax.set_ylabel('$z$-dependent frequency component [m]', size=12)

plt.tight_layout()
plt.savefig('z.png', dpi=200, bbox_inches='tight')
