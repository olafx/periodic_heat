#pragma once
#include <array>
#include <complex>
#include <fftw3.h>

using std::array, std::complex;

void linear_fill(double *data, size_t n, double begin, double step)
{   for (size_t i = 0; i < n; i++)
    {   data[i] = begin + step * i;
    }
}



void eval_coords_real_space(double *x, double *y, array<size_t, 2> n, array<double, 4> range)
{   linear_fill(x, n[0], range[0], (range[1] - range[0]) / n[0]);
    linear_fill(y, n[1], range[2], (range[3] - range[2]) / n[1]);
}

void eval_coords_freq_space(double *k_x, double *k_y, array<size_t, 2> n, array<double, 4> range)
{   linear_fill(k_x, n[0], 0,                                        1 / (range[1] - range[0]));
    linear_fill(k_y, n[1], -.5 * (n[1] - 1) / (range[3] - range[2]), 1 / (range[3] - range[2]));
}



void eval_intensity_real_space(complex<double> *q, array<size_t, 2> n, double *x, double *y,
                               double q_P, array<double, 2> q_center, double q_sd)
{   double a = .5 * M_1_PI * q_P / (q_sd * q_sd);
    for (size_t j = 0; j < n[1]; j++)
        for (size_t i = 0; i < n[0]; i++)
        {   double b1 = (x[i] - q_center[0]) / q_sd;
            double b2 = (y[j] - q_center[1]) / q_sd;
            q[i + j * n[0]] = a * exp(-.5 * (b1 * b1 + b2 * b2));
        }
}

void eval_intensity_freq_space(complex<double> *q_dft, array<size_t, 2> n, complex<double> *q)
{   fftw_plan plan = fftw_plan_dft_2d(n[0], n[1],
                                      reinterpret_cast<fftw_complex *>(q), reinterpret_cast<fftw_complex *>(q_dft),
                                      FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    for (size_t i = 0; i < n[0] * n[1]; i++)
        q_dft[i] /= n[0] * n[1];
}



void eval_sigma_0(complex<double> *s, array<size_t, 2> n, double *k_x, double *k_y, array<double, 3> a)
{   for (size_t j = 0; j < n[1]; j++)
        for (size_t i = 0; i < n[0]; i++)
            s[i + j * n[0]] = 2 * M_PI * sqrt((a[0] * (k_x[i] * k_x[i])
                                             + a[1] * (k_y[j] * k_y[j])) / a[2]);
}

void eval_sigma_1(complex<double> *s, array<size_t, 2> n, double *k_x, double *k_y, array<double, 3> a, double q_p)
{   using namespace std::complex_literals;
    for (size_t j = 0; j < n[1]; j++)
        for (size_t i = 0; i < n[0]; i++)
            s[i + j * n[0]] = sqrt(2i * M_PI) * sqrt((1i / q_p + 2 * M_PI * a[0] * k_x[i] * k_x[i]
                                                               + 2 * M_PI * a[1] * k_y[j] * k_y[j]) / a[2]);
}



void eval_time_mutual_T_freq_space(complex<double> *m_T_dft, array<size_t, 2> n, complex<double> *q_dft, complex<double> *s,
                                   double b_z, double z, double d)
{   for (size_t i = 0; i < n[0] * n[1]; i++)
    {   complex<double> a = q_dft[i] / (b_z * s[i]),
                        b = z * s[i],
                        c  = 2 * d * s[i];
        //  the below approximates a * (exp(b) / (1. - exp(c)) - exp(-b) / (1. - exp(-c))), which is too unstable due to large c
        m_T_dft[i] = a * (-exp(b - c) - exp(-b) / (1. - exp(-c)));
    }
}

void eval_time_mutual_T_real_space(complex<double> *m_T, complex<double> *m_T_dft, array<size_t, 2> n)
{   fftw_plan plan = fftw_plan_dft_2d(n[1], n[0],
                                      reinterpret_cast<fftw_complex *>(m_T_dft), reinterpret_cast<fftw_complex *>(m_T),
                                      FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    for (size_t i = 0; i < n[0] * n[1]; i++)
        m_T[i] *= n[0] * n[1];
}



void eval_T(complex<double> *T, complex<double> *m_T, array<size_t, 2> n, double q_p, double t)
{   using namespace std::complex_literals;
    complex<double> *m_T_0 = m_T,
                    *m_T_1 = m_T + n[0] * n[1];
    for (size_t i = 0; i < n[0] * n[1]; i++)
        T[i] = .5 * (m_T_0[i] + m_T_1[i] * exp(2i * M_PI * t / q_p));
}
