#include "solver_sine.hh"
#include "storage.hh"

int main()
{
    //  independent settings

    std::array<size_t, 2> n     {1000, 1000};       //  real space dims
    std::array<double, 3> a     {3e-7, 1e-7, 1e-7}; //  heat diffusivities (x, y, z)
    std::array<double, 4> range {0, .1, 0, .1};     //  real space range (x, y)
    double b_z   = .1,   //  heat conductivity along z (related to but not independent of a[2])
           d     = 2e-3, //  depth; real space z range is [0, d)
           z_rel = .02,  //  evaluation relative depth
           q_P   = .5,   //  power (area integrated intensity) of laser
           q_p   = 10,   //  period of sinusoidal laser intensity
           q_sd  = 1e-3, //  real space standard deviation of Gaussian laser dot
           t     = 7;    //  evaluation time

    //  dependent settings

    std::array<size_t, 2> n_dft {n[0] / 2 + 1, n[1] / 2 + 1}; //  quarter frequency space dims
    double z = z_rel * d; //  evaluation depth
    std::array<double, 2> range_size {range[1] - range[0], range[3] - range[2]},               //  size of range
                          q_center   {.5 * (range[0] + range[1]), .5 * (range[2] + range[3])}; //  laser dot location (x, y)



    //  space and frequency space coordinates

    double *x   = new double[n[0]], *y   = new double[n[1]],
           *k_x = new double[n[0]], *k_y = new double[n[1]];
    linear_fill(x, n[0], range[0], range_size[0] / n[0]);
    linear_fill(y, n[1], range[2], range_size[1] / n[1]);
    //  probably needs to change. not only does this assume shifted, it also doesn't account for r2c rather than c2c
    linear_fill(k_x, n[0], -.5 * (n[0] - 1) / range[0], 1 / range_size[0]);
    linear_fill(k_y, n[1], -.5 * (n[1] - 1) / range[2], 1 / range_size[1]);



    //  real space Gaussian intensity

    double *q = new double[n[0] * n[1]];
    gaussian_intensity(q, n, x, y, q_P, q_center, q_sd);

    //  frequency space Gaussian intensity (via real to complex quarter FFT)

    //  real to complex quarter FFT has n[0] / 2 + 1 by n[1] / 2 + 1 output
    std::complex<double> *q_dft = new std::complex<double>[n_dft[0] * n_dft[1] + 250000]; //  TODO padding
    fftw_plan plan = fftw_plan_dft_r2c_2d(n[0], n[1], q, reinterpret_cast<fftw_complex *>(q_dft), FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    /*
    plan = fftw_plan_dft_c2r_2d(n[0], n[1], reinterpret_cast<fftw_complex *>(q_dft), q, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    */
    //  renormalization
    for (size_t i = 0; i < n[0] * n[1]; i++)
        q[i] /= n[0] * n[1];



    //  sigma for frequencies 0 and 1 / q_p

    //  s storage format:
    //      freq 0       pos k_x[0] k_y[0],        ..., freq 0       pos k_x[n[0] / 2] k_y[0],
    //      ...,
    //      freq 0       pos k_x[0] k_y[n[1] / 2], ..., freq 0       pos k_x[n[0] / 2] k_y[n[1] / 2],
    //      freq 1 / q_p pos k_x[0] k_y[0],        ..., freq 1 / q_p pos k_x[n[0] / 2] k_y[0],
    //      ...,
    //      freq 1 / q_p pos k_x[0] k_y[n[1] / 2], ..., freq 1 / q_p pos k_x[n[0] / 2] k_y[n[1] / 2],
    std::complex<double> *s = new std::complex<double>[n_dft[0] * n_dft[1] * 2];
    s_0(s                      , n_dft, k_x, k_y, a);
    s_1(s + n_dft[0] * n_dft[1], n_dft, k_x, k_y, a, q_p);

    store(static_cast<void *>(q_dft), {n_dft[0], n_dft[1], 1}, "0.vti");
}

//  the q_dft array should be of size n_dft[0] * n_dft[1] instead, but then it segfaults. this is probably due to padding
//  required by FFTW. need to read documentation more. seems like rather than 1/4 memory, it needs 1/2.
//  don't worry about this being too much space or whatever; get the project to work first, then optimize, because for example
//  there's also things like in place FFT that would be even better.
