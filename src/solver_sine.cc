#include "solver_sine.hh"
#include "storage.hh"

int main()
{
    //  independent settings

    array<size_t, 2> n     {1000, 1000};       //  real space dims
    array<double, 3> a     {3e-7, 1e-7, 1e-7}; //  heat diffusivities (x, y, z)
    array<double, 4> range {0, .1, 0, .1};     //  real space range (x, y)
    double b_z   = .1,   //  heat conductivity along z (related to but not independent of a[2])
           d     = 2e-3, //  depth; real space z range is [0, d)
           z_rel = .02,  //  evaluation relative depth
           q_P   = .5,   //  power (area integrated intensity) of laser
           q_p   = 10,   //  period of sinusoidal laser intensity
           q_sd  = 1e-3, //  real space standard deviation of Gaussian laser dot
           t     = 7;    //  evaluation time

    //  dependent settings

    double z = z_rel * d; //  evaluation depth
    array<double, 2> q_center {.5 * (range[0] + range[1]), .5 * (range[2] + range[3])}; //  laser dot location (x, y)



    //  space and frequency space coordinates

    double *x   = new double[n[0]], *y   = new double[n[1]],
           *k_x = new double[n[0]], *k_y = new double[n[1]];
    eval_coords_real_space(x,   y,   n, range);
    eval_coords_freq_space(k_x, k_y, n, range);



    //  real space Gaussian intensity

    complex<double> *q = new complex<double>[n[0] * n[1]];
    eval_intensity_real_space(q, n, x, y, q_P, q_center, q_sd);
    delete[] x;
    delete[] y;

    //  frequency space Gaussian intensity

    complex<double> *q_dft = new complex<double>[n[0] * n[1]];
    eval_intensity_freq_space(q_dft, n, q);
    delete[] q;



    //  sigma for frequencies 0 and 1 / q_p

    //  s storage format:
    //      freq 0       pos k_x[0] k_y[0],        ..., freq 0       pos k_x[n[0] - 1] k_y[0],
    //      ...,
    //      freq 0       pos k_x[0] k_y[n[1] - 1], ..., freq 0       pos k_x[n[0] - 1] k_y[n[1] - 1],
    //      freq 1 / q_p pos k_x[0] k_y[0],        ..., freq 1 / q_p pos k_x[n[0] - 1] k_y[0],
    //      ...,
    //      freq 1 / q_p pos k_x[0] k_y[n[1] - 1], ..., freq 1 / q_p pos k_x[n[0] - 1] k_y[n[1] - 1]
    complex<double> *s = new complex<double>[n[0] * n[1] * 2];
    eval_sigma_0(s              , n, k_x, k_y, a);
    eval_sigma_1(s + n[0] * n[1], n, k_x, k_y, a, q_p);
    delete[] k_x;
    delete[] k_y;



    //  time mutual T components for frequencies 0 and 1 / q_p in frequency space

    //  m_T_dft storage format:
    //      freq 0       pos k_x[0] k_y[0],        ..., freq 0       pos k_x[n[0] - 1] k_y[0],
    //      ...,
    //      freq 0       pos k_x[0] k_y[n[1] - 1], ..., freq 0       pos k_x[n[0] - 1] k_y[n[1] - 1],
    //      freq 1 / q_p pos k_x[0] k_y[0],        ..., freq 1 / q_p pos k_x[n[0] - 1] k_y[0],
    //      ...,
    //      freq 1 / q_p pos k_x[0] k_y[n[1] - 1], ..., freq 1 / q_p pos k_x[n[0] - 1] k_y[n[1] - 1]
    complex<double> *m_T_dft = new complex<double>[n[0] * n[1] * 2];
    eval_time_mutual_T_freq_space(m_T_dft,               n, q_dft, s,               b_z, z, d);
    eval_time_mutual_T_freq_space(m_T_dft + n[0] * n[1], n, q_dft, s + n[0] * n[1], b_z, z, d);
    delete[] s;
    delete[] q_dft;



    //  time mutual T components for frequencies 0 and 1 / q_p in real space

    //  m_T storage format:
    //      freq 0       pos x[0] y[0],        ..., freq 0       pos x[n[0] - 1] y[0],
    //      ...,
    //      freq 0       pos x[0] y[n[1] - 1], ..., freq 0       pos x[n[0] - 1] y[n[1] - 1],
    //      freq 1 / q_p pos x[0] y[0],        ..., freq 1 / q_p pos x[n[0] - 1] y[0],
    //      ...,
    //      freq 1 / q_p pos x[0] y[n[1] - 1], ..., freq 1 / q_p pos x[n[0] - 1] y[n[1] - 1]
    complex<double> *m_T = new complex<double>[n[0] * n[1] * 2];
    eval_time_mutual_T_real_space(m_T,               m_T_dft,               n);
    eval_time_mutual_T_real_space(m_T + n[0] * n[1], m_T_dft + n[0] * n[1], n);
    delete[] m_T_dft;



    //  T in real space

    complex<double> *T = new complex<double>[n[0] * n[1]];
    eval_T(T, m_T, n, q_p, t);
    delete[] m_T;



    //  storage

    store(static_cast<void *>(T), {n[0], n[1], 1}, "0.vti");
}
