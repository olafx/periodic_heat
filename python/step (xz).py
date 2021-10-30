import numpy as np
import matplotlib.pyplot as plt

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
y_p = y_n//2
z_n = 100
d = 2e-3
y = 0
Q_P = 0.5
Q_p = 10.0
t = 7.0
Q_c_x = (x_b-x_a)/2
Q_c_y = (y_b-y_a)/2
Q_s_x = 1e-3
Q_s_y = 1e-3
dpi = 100
m = 60
USE_LONG_DOUBLE = False
DEBUG = False
CONTOUR = False



x, y = np.meshgrid(
    np.linspace(x_a, x_b, x_n), 
    np.linspace(y_a, y_b, y_n))
z = np.linspace(0, d, z_n)
print('computed space')
k_x, k_y = np.meshgrid(
    np.linspace(-0.5*(x_n-1)/(x_b-x_a), 0.5*(x_n-1)/(x_b-x_a), x_n),
    np.linspace(-0.5*(y_n-1)/(y_b-y_a), 0.5*(y_n-1)/(y_b-y_a), y_n))
print('computed freq space')

Q = Q_P/(2*np.pi*Q_s_x*Q_s_y)*np.exp(-0.5*(((x-Q_c_x)/Q_s_x)**2+((y-Q_c_y)/Q_s_y)**2))
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



T_xz = np.empty([z_n, x_n])
for zi in range(0, z_n):

    zz = z[zi]
    print('computing for z '+str(zz))

    if USE_LONG_DOUBLE:
        T_f1 = np.empty([x_n, y_n, 2*m+1], dtype='complex256')
    else:
        T_f1 = np.empty([x_n, y_n, 2*m+1], dtype='complex128')
    T_f1[:,:,0] = Q_F_yx/(s[:,:,0]*b_z)*(np.exp(s[:,:,0]*zz)/(1-np.exp(2*s[:,:,0]*d))-np.exp(-s[:,:,0]*zz)/(1-np.exp(-2*s[:,:,0]*d)))
    print('computed T component (freq space) for freq 0')
    for ki in range(1, 2*m+1):
        k = 2*(ki-m)-1
        T_f1[:,:,ki] = Q_F_yx/(s[:,:,ki]*b_z)*(np.exp(s[:,:,ki]*zz)/(1-np.exp(2*s[:,:,ki]*d))-np.exp(-s[:,:,ki]*zz)/(1-np.exp(-2*s[:,:,ki]*d)))
        print('computed T component for current z (freq space) for freq with k '+str(k))

    if DEBUG: print('max abs of T component for current z at any freq (freq space): '+str(np.max(np.abs(T_f1))))

    T_1 = np.fft.ifft(np.fft.ifft(np.fft.fftshift(T_f1)*y_n,y_n,1)*x_n,x_n,0)
    print('computed T components for current z (real space)')
    del T_f1

    T = 0.5*Q_P*T_1[:,:,0]
    for ki in range(1, 2*m+1):
        k = 2*(ki-m)-1
        T += (1j*Q_P)/(k*np.pi)*T_1[:,:,ki]*np.exp(2j*np.pi*k*t/Q_p)
    del T_1
    T = np.abs(T)
    print('T computed for current z')

    T_xz[zi,:] = T[:,y_p]
    del T
    print('added T for current z to data')    

del s
del Q_F_yx



print('T computed for all z')



plt.rcParams['font.family'] = 'SF Mono'



fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)

if CONTOUR:
    img = ax.contour(1000*np.linspace(x_a, x_b, x_n), 1000*z, T_xz, levels=[40, 60, 80, 100, 200])
    ax.clabel(img, fontsize=10)
else:
    img = ax.pcolormesh(1000*np.linspace(x_a, x_b, x_n), 1000*z, T_xz, shading='auto')
    cb = fig.colorbar(img)
del T_xz

ax.set_title('Time $t$ '+str(round(t*1000)).zfill(4)+' [ms]')
ax.set_xlabel('$x$ [mm]')
ax.set_ylabel('$z$ [mm]')

if not CONTOUR:
    cb.set_label('Temperature $T$ [K]')

print('T plotted')

plt.savefig('T (xz).png', dpi=dpi, bbox_inches='tight')
print('saved T depth (graph)')
