# periodic_heat

Semi analytical anisotropic heat solver for thin materials radiated periodically by a Gaussian intensity (laser).

A usual 3D heat solver uses 6 FFTs. This one uses 2. Only the x and y dimensions need to be inverse transformed, that's it. The
x and y dimensions have analytic forward Fourier transforms. The z dimension is evaluated fully analytically based on analytic
frequency decomposition of the periodic intensity. This allows the temperature field at any depth to be evaluated using only 2D
memory, giving this method O(n^2) memory complexity for an n by n by n grid.

This code stores a 3D temperature field in a VTK image file by looping over various z. A single z evaluation is a full
solution to the 3D heat equation however; it only loops to obtain 3D data.

Not everything about this solver works perfectly the way it should, as described in the thesis (KU Leuven undergrad physics).

Some old python code is included too, which has more functionality but is less refined.

## Renders

Sinusoidal intensity.

![sine 1](https://raw.githubusercontent.com/olafx/periodic_heat/master/renders/sine%201.png)
![sine 2](https://raw.githubusercontent.com/olafx/periodic_heat/master/renders/sine%202.png)

Periodic heaviside intensity.

![step 1](https://raw.githubusercontent.com/olafx/periodic_heat/master/renders/step%201.png)
