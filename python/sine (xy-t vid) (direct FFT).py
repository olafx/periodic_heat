import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a_x = 3e-7
a_y = 1e-7
a_z = 1e-7
b_z = .1
x_a = .0
x_b = .1
x_n = 400
y_a = .0
y_b = .1
y_n = 400
d = 2e-3
z = 0.02*d
Q_P = 0.5
Q_p = 10.0
Q_c_x = (x_b-x_a)/2
Q_c_y = (y_b-y_a)/2
Q_s_x = 1e-3
Q_s_y = 1e-3
t_i = 0.0
t_f = t_i + Q_p
t_n = 400
t_n_plot = 40
fps = t_n/(t_f-t_i)
dpi = 200
USE_LONG_DOUBLE = False
DEBUG = False
CONTOUR = True



x, y, t = np.meshgrid(
    np.linspace(x_a, x_b, x_n),
    np.linspace(y_a, y_b, y_n),
    np.linspace(t_i, t_f-1/fps, t_n))
print('computed space and time')
k_x, k_y, f = np.meshgrid(
    np.linspace(-0.5*(x_n-1)/(x_b-x_a), 0.5*(x_n-1)/(x_b-x_a), x_n),
    np.linspace(-0.5*(y_n-1)/(y_b-y_a), 0.5*(y_n-1)/(y_b-y_a), y_n),
    np.linspace(-0.5*(t_n-1)/(t_f-t_i), 0.5*(t_n-1)/(t_f-t_i), t_n))
print('computed freq space and time')



Q = Q_P/(2*np.pi*Q_s_x*Q_s_y)*np.exp(-0.5*(((x-Q_c_x)/Q_s_x)**2+((y-Q_c_y)/Q_s_y)**2))
t_temp = np.linspace(t_i, t_f-1/fps, t_n)
for i in range(0, t_n):
    if np.mod(2*t_temp[i]/Q_p,t_n) < 1:
        Q[:,:,i] = 0
del t_temp

print('computed intensity (real space)')
print(np.shape(Q))

Q_F_yxt = np.fft.fftshift(np.fft.fftn(Q)/t_n/x_n/y_n)
print('computed intensity (freq space)')
print(np.shape(Q_F_yxt))

if DEBUG:
    print('max intensity (real space): '+str(np.max(Q)))
    print('max intensity (freq space): '+str(np.max(Q_F_yxt)))
del Q



if USE_LONG_DOUBLE:
    s = np.empty([y_n, x_n, t_n], dtype='complex256')
else:
    s = np.empty([y_n, x_n, t_n], dtype='complex128')
s[:,:,:] = np.sqrt(2j*np.pi)*np.sqrt((1j*f/Q_p+2*np.pi*a_x*k_x**2+2*np.pi*a_y*k_y**2)/a_z)
print('computed σ (freq space)')
print(np.shape(s))

if DEBUG:
    s_max = np.max(np.real(s))
    print('max real(σ): '+str(s_max))
    print('max real(σ*z): '+str(s_max*z))
    print('max real(2*σ*d): '+str(2*s_max*d))
    del s_max



if USE_LONG_DOUBLE:
    T_f = np.empty([y_n, x_n, t_n], dtype='complex256')
else:
    T_f = np.empty([y_n, x_n, t_n], dtype='complex128')
T_f[:,:,:] = Q_F_yxt/(s[:,:,:]*b_z)*(np.exp(s[:,:,:]*z)/(1-np.exp(2*s[:,:,:]*d))-np.exp(-s[:,:,:]*z)/(1-np.exp(-2*s[:,:,:]*d)))
print('computed T component (freq space)')
del s
del Q_F_yxt
print(np.shape(T_f))

if DEBUG: print('max abs of T component at either freq (freq space): '+str(np.max(np.abs(T_f))))



T = np.fft.ifftn(np.fft.fftshift(T_f)*t_n*x_n*y_n)
print('computed T components (real space)')
del T_f
print(np.shape(T))

if DEBUG: print('max abs of T component at any freq (freq space): '+str(np.max(np.abs(T))))



plt.rcParams['font.family'] = 'SF Mono'



fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)



p_T = []
p_t = []
x = np.linspace(x_a, x_b, x_n)
y = np.linspace(y_a, y_b, y_n)
t = np.linspace(t_i, t_f-1/fps, t_n)



def plot1(ti):

    tt = t[ti]
    p_t.append(tt)

    print('time '+str(tt))

    if not CONTOUR and len(plt.gca().collections) != 0:
        plt.gca().collections[-1].colorbar.remove()

    plt.cla()

    p_T.append(np.real(T[x_n//2,y_n//2,ti]))

    if CONTOUR:
        img = ax.contour(x*1000, y*1000, np.real(T[:,:,ti]), levels=[0.5])
        ax.clabel(img, fontsize=10)
    else:
        img = ax.pcolormesh(x*1000, y*1000, np.abs(T[:,:,ti]), shading='auto')
        cb = fig.colorbar(img)

    ax.set_title('Time $t$ '+str(round(tt*1000)).zfill(4)+' [ms]')
    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$y$ [mm]')
    ax.set_xlim([20, 80])
    ax.set_ylim([20, 80])

    if not CONTOUR:
        cb.set_label('Temperature $T$ [K]')

    print('\tT plotted')

    return img, ax



vid1 = FuncAnimation(fig, plot1, frames=np.arange(0,t_n,t_n//t_n_plot), interval=200/fps)
vid1.save('T main.gif', dpi=dpi)
print('saved T over time (field) (video)')



p_T.pop(0)
p_t.pop(0)

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)

ax.plot(p_t, np.abs(p_T))

plt.savefig('T point.png', dpi=dpi, bbox_inches='tight')
print('saved T over time (center only) (graph)')
