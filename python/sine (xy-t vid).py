import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a_x = 3e-7
a_y = 1e-7
a_z = 1e-7
b_z = .1
x_a = .0
x_b = .1
x_n = 1000
y_a = .0
y_b = .1
y_n = 1000
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
t_n = 20
fps = t_n/(t_f-t_i)
dpi = 100
USE_LONG_DOUBLE = False
DEBUG = True
CONTOUR = False
DERIVATIVE = False
NORMALIZING = False



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



Q = Q_P/(2*np.pi*Q_s_x*Q_s_y)*np.exp(-0.5*(((x-Q_c_x)/Q_s_x)**2+((y-Q_c_y)/Q_s_y)**2))
print('computed intensity (real space)')

Q_F_yx = np.fft.fftshift(np.fft.fft(np.fft.fft(Q,x_n,0)/x_n,y_n,1)/y_n)
print('computed intensity (freq space)')

if DEBUG:
    print('max intensity (real space): '+str(np.max(Q)))
    print('max intensity (freq space): '+str(np.max(Q_F_yx)))
del Q





# plt.pcolormesh(np.abs(Q_F_yx[:,:]))
# plt.colorbar()
# plt.show()
# exit()




if USE_LONG_DOUBLE:
    s = np.empty([x_n, y_n, 2], dtype='complex256')
else:
    s = np.empty([x_n, y_n, 2], dtype='complex128')
s[:,:,0] = 2*np.pi*np.sqrt((a_x*k_x**2+a_y*k_y**2)/a_z)
print('computed σ for freq 0')
s[:,:,1] = np.sqrt(2j*np.pi)*np.sqrt((1j/Q_p+2*np.pi*a_x*k_x**2+2*np.pi*a_y*k_y**2)/a_z)
print('computed σ for freq inverse of given period')

if DEBUG:
    s_max = np.max(np.real(s))
    print('max real(σ): '+str(s_max))
    print('max real(σ*z): '+str(s_max*z))
    print('max real(2*σ*d): '+str(2*s_max*d))
    del s_max

# temp
# fig = plt.figure(figsize=(5,4))
# ax = fig.add_subplot(111)

# img = ax.pcolormesh(x*1000, y*1000, np.real(2*s[:,:,0]*d))
# cb = plt.colorbar(img)
# cb.set_label('$2 \sigma d$ [ ]')

# ax.set_xlabel('$x$ [mm]')
# ax.set_ylabel('$y$ [mm]')

# plt.savefig('2 sigma d.png', dpi=200, bbox_inches='tight')
# temp





# plt.pcolormesh(np.real(s[:,:,1]))
# plt.colorbar()
# plt.show()
# exit()







if USE_LONG_DOUBLE:
    T_f1 = np.empty([x_n, y_n, 2], dtype='complex256')
else:
    T_f1 = np.empty([x_n, y_n, 2], dtype='complex128')

T_f1[:,:,0] = Q_F_yx/(s[:,:,0]*b_z)*(np.exp(s[:,:,0]*z)/(1-np.exp(2*s[:,:,0]*d))-np.exp(-s[:,:,0]*z)/(1-np.exp(-2*s[:,:,0]*d)))
print('computed T component (freq space) for freq 0')
T_f1[:,:,1] = Q_F_yx/(s[:,:,1]*b_z)*(np.exp(s[:,:,1]*z)/(1-np.exp(2*s[:,:,1]*d))-np.exp(-s[:,:,1]*z)/(1-np.exp(-2*s[:,:,1]*d)))
print('computed T component (freq space) for freq inverse of given period')
del s
del Q_F_yx

if DEBUG: print('max abs of T component at either freq (freq space): '+str(np.max(np.abs(T_f1))))




# plt.pcolormesh(np.real(T_f1[:,:,0]))
# plt.colorbar()
# plt.show()
# exit()




T_1 = np.fft.ifft(np.fft.ifft(np.fft.fftshift(T_f1)*y_n,y_n,1)*x_n,x_n,0)
print('computed T components (real space)')
del T_f1

if DEBUG: print('max abs of T component at any freq (freq space): '+str(np.max(np.abs(T_1))))



if NORMALIZING:
    T_1[:,:,0] /= (np.max(T_1[:,:,0])-np.min(T_1[:,:,0]))
    T_1[:,:,1] /= 2*(np.max(T_1[:,:,1])-np.min(T_1[:,:,1]))



print('time-independent math done')



plt.rcParams['font.family'] = 'SF Mono'



fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)



p_T = []



def plot1(t):
    print('time '+str(t))

    if not CONTOUR and len(plt.gca().collections) != 0:
        plt.gca().collections[-1].colorbar.remove()

    plt.cla()

    T = 0.5*Q_P*(T_1[:,:,0]+T_1[:,:,1]*np.exp(2j*np.pi*t/Q_p))
    if DERIVATIVE:
        T -= np.abs(0.5*Q_P*(T_1[:,:,0]+T_1[:,:,1]*np.exp(2j*np.pi*(t-1/fps)/Q_p)))
        T *= fps
    p_T.append(np.real(T[x_n//2,y_n//2]))
    print('\tT computed')

    if CONTOUR:
        img = ax.contour(x*1000, y*1000, T, levels=[40, 60, 80, 100, 200])
        ax.clabel(img, fontsize=10)
    else:
        img = ax.pcolormesh(x*1000, y*1000, np.abs(T), shading='auto')
        cb = fig.colorbar(img)
    del T

    ax.set_title('Time $t$ '+str(round(t*1000)).zfill(4)+' [ms]')
    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$y$ [mm]')

    if not CONTOUR:
        cb.set_label('Temperature $T$ [K]')

    print('\tT plotted')

    return img, ax



vid1 = FuncAnimation(fig, plot1, frames=t, interval=200/fps)
vid1.save('T main.gif', dpi=dpi)
print('saved T over time (field) (video)')



p_T.pop(0)

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)

ax.plot(t, p_T)

plt.savefig('T point.png', dpi=dpi, bbox_inches='tight')
print('saved T over time (center only) (graph)')
