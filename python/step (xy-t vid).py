import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a_x = 3e-7            # diffusivities x, y, z
a_y = 1e-7
a_z = 1e-7

b_z = .1              # conductivity z
x_a = .0              # x domain
x_b = .1
x_n = 1600

y_a = .0              # y domain
y_b = .1
y_n = 1600

d = 2e-3              # thickness

z = 0.02*d            # z

Q_P = 0.5             # LASER power
Q_p = 10.0            # LASER period
Q_c_x = (x_b-x_a)/2   # LASER Gaussian mean
Q_c_y = (y_b-y_a)/2
Q_s_x = 1e-3          # LASER Gaussian sd
Q_s_y = 1e-3

t_i = 0.0             # t domain
t_f = t_i + Q_p
t_n = 20

fps = t_n/(t_f-t_i)   # Framerate (for gif)
dpi = 200             # Image dpi

m = 40                # Number of positive frequencies
                      # used to approximate the step

USE_LONG_DOUBLE = False    # Use higher precision floating point numbers, for instability debugging
DEBUG = False              # Prints intermediate extreme values of tensors, for instability debugging
CONTOUR = False            # Plots contours instead of colormesh
DERIVATIVE = False         # Uses time-derivative of T instead of t in plots



x, y = np.meshgrid(
    np.linspace(x_a, x_b, x_n), 
    np.linspace(y_a, y_b, y_n))
print('computed space')
k_x, k_y = np.meshgrid(
    np.linspace(-0.5*(x_n-1)/(x_b-x_a), 0.5*(x_n-1)/(x_b-x_a), x_n),
    np.linspace(-0.5*(y_n-1)/(y_b-y_a), 0.5*(y_n-1)/(y_b-y_a), y_n))
print('computed freq space')
t = np.linspace(t_i, t_f-1/fps, t_n)
print('computed time')

Q = 1/(2*np.pi*Q_s_x*Q_s_y)*np.exp(-0.5*(((x-Q_c_x)/Q_s_x)**2+((y-Q_c_y)/Q_s_y)**2))
print('computed intensity (real space)')

Q_F_yx = np.fft.fftshift(np.fft.fft(np.fft.fft(Q,x_n,0)/x_n,y_n,1)/y_n)
print('computed intensity (freq space)')

if DEBUG:
    print('max intensity (real space): '+str(np.max(Q)))
    print('max intensity (freq space): '+str(np.max(Q_F_yx)))
del Q


if USE_LONG_DOUBLE:
    s = np.empty([x_n, y_n, 2*m+1], dtype='complex256')
else:
    s = np.empty([x_n, y_n, 2*m+1], dtype='complex128')
s[:,:,0] = 2*np.pi*np.sqrt((a_x*k_x**2+a_y*k_y**2)/a_z)
print('computed σ for freq 0')
for ki in range(1,2*m+1):
    k = 2*(ki-m)-1
    s[:,:,ki] = np.sqrt(2*np.pi)*np.sqrt((1j*k/Q_p+2*np.pi*a_x*k_x**2+2*np.pi*a_y*k_y**2)/a_z)
    print('computed σ for freq with k '+str(k))

if DEBUG:
    s_max = np.max(np.real(s))
    print('max real(σ): '+str(s_max))
    print('max real(σ*z): '+str(s_max*z))
    print('max real(2*σ*d): '+str(2*s_max*d))
    del s_max


if USE_LONG_DOUBLE:
    T_f1 = np.empty([x_n, y_n, 2*m+1], dtype='complex256')
else:
    T_f1 = np.empty([x_n, y_n, 2*m+1], dtype='complex128')
T_f1[:,:,0] = Q_F_yx/(s[:,:,0]*b_z)*(np.exp(s[:,:,0]*z)/(1-np.exp(2*s[:,:,0]*d))-np.exp(-s[:,:,0]*z)/(1-np.exp(-2*s[:,:,0]*d)))
print('computed T component (freq space) for freq 0')
for ki in range(1, 2*m+1):
    k = 2*(ki-m)-1
    T_f1[:,:,ki] = Q_F_yx/(s[:,:,ki]*b_z)*(np.exp(s[:,:,ki]*z)/(1-np.exp(2*s[:,:,ki]*d))-np.exp(-s[:,:,ki]*z)/(1-np.exp(-2*s[:,:,ki]*d)))
    print('computed T component (freq space) for freq with k '+str(k))
del s
del Q_F_yx

if DEBUG: print('max abs of T component at any freq (freq space): '+str(np.abs(np.max(T_f1))))



T_1 = np.fft.ifft(np.fft.ifft(np.fft.fftshift(T_f1)*y_n,y_n,1)*x_n,x_n,0)
print('computed T components (real space)')
del T_f1

if DEBUG: print('max abs of T component at any freq (real space): '+str(np.max(np.abs(T_1))))



print('time-independent math done')



fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)



p_T = []



def plot1(t):
    print('time '+str(t))

    if not CONTOUR and len(plt.gca().collections) != 0:
        plt.gca().collections[-1].colorbar.remove()

    plt.cla()

    T = 0.5*Q_P*T_1[:,:,0]

    for ki in range(1, 2*m+1):
        k = 2*(ki-m)-1
        T += (1j*Q_P)/(k*np.pi)*T_1[:,:,ki]*np.exp(2j*np.pi*k*t/Q_p)
    
    T = np.abs(T)
    
    if DERIVATIVE:
        T_old = 0.5*Q_P*T_1[:,:,0]
        for ki in range(1, 2*m+1):
            k = 2*(ki-m)-1
            T_old += (1j*Q_P)/(k*np.pi)*T_1[:,:,ki]*np.exp(2j*np.pi*k*(t-1/fps)/Q_p)
        T -= np.abs(T_old)    
        T *= fps
    p_T.append(T[x_n//2,y_n//2])
    print('\tT computed')

    if CONTOUR:
        img = ax.contour(x*1000, y*1000, T, levels=[40, 60, 80, 100, 200])
        ax.clabel(img, fontsize=10)
    else:
        img = ax.pcolormesh(x*1000, y*1000, np.log(1+T), shading='auto')
        cb = fig.colorbar(img)
    del T

    ax.set_title('Time $t$ '+str(round(t*1000)).zfill(4)+' [ms]')
    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$y$ [mm]')

    if not CONTOUR:
        cb.set_label('Temperature $log_{10}(1+T)$ [K]')

    print('\tT plotted')

    return img, ax



vid1 = FuncAnimation(fig, plot1, frames=t, interval=200/fps)
vid1.save('T.gif', dpi=dpi)
print('saved T over time (field) (video)')



p_T.pop(0)

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)

ax.plot(t, p_T, c='black')

ax.set_xlabel('$t$ [s]')
ax.set_ylabel('$T$ [K]')
ax.set_xlim([t_i,t_f])

plt.savefig('T.png', dpi=dpi, bbox_inches='tight')
print('saved T over time (center only) (graph)')
