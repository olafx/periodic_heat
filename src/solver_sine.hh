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



void eval_intensity_freq_space(complex<double> *q_ft, array<size_t, 3> n, double *k_x, double *k_y, double q_P,
                               array<double, 2> q_center, double q_sd)
{   using namespace std::complex_literals;
    for (size_t j = 0; j < n[1]; j++)
        for (size_t i = 0; i < n[0]; i++)
            q_ft[i + j * n[0]] = q_P * exp(-2 * M_PI * M_PI * q_sd * q_sd * (k_x[i] * k_x[i] + k_y[j] * k_y[j]))
                                     * exp(-2i * M_PI * (q_center[0] * k_x[i] + q_center[1] * k_y[j]));
}



void eval_sigma_0(complex<double> *s, array<size_t, 3> n, double *k_x, double *k_y, array<double, 3> a)
{   for (size_t j = 0; j < n[1]; j++)
        for (size_t i = 0; i < n[0]; i++)
            s[i + j * n[0]] = 2 * M_PI * sqrt((a[0] * (k_x[i] * k_x[i])
                                             + a[1] * (k_y[j] * k_y[j])) / a[2]);
}

void eval_sigma_1(complex<double> *s, array<size_t, 3> n, double *k_x, double *k_y, array<double, 3> a, double q_p)
{   using namespace std::complex_literals;
    for (size_t j = 0; j < n[1]; j++)
        for (size_t i = 0; i < n[0]; i++)
            s[i + j * n[0]] = sqrt(2i * M_PI) * sqrt((1i / q_p + 2 * M_PI * a[0] * k_x[i] * k_x[i]
                                                               + 2 * M_PI * a[1] * k_y[j] * k_y[j]) / a[2]);
}



void eval_time_mutual_T_freq_space(complex<double> *m_T_dft, array<size_t, 3> n, complex<double> *q_dft, complex<double> *s,
                                   double b_z, double z, double d)
{   for (size_t i = 0; i < n[0] * n[1]; i++)
    {   complex<double> a = q_dft[i] / (b_z * s[i]),
                        b = z * s[i],
                        c  = 2 * d * s[i];
        //  the below approximates a * (exp(b) / (1. - exp(c)) - exp(-b) / (1. - exp(-c))), which is too unstable due to large c
        m_T_dft[i] = a * (-exp(b - c) - exp(-b) / (1. - exp(-c)));
    }
}

void eval_time_mutual_T_real_space(complex<double> *m_T, complex<double> *m_T_dft, array<size_t, 3> n)
{   fftw_plan plan = fftw_plan_dft_2d(n[1], n[0],
                                      reinterpret_cast<fftw_complex *>(m_T_dft), reinterpret_cast<fftw_complex *>(m_T),
                                      FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    for (size_t i = 0; i < n[0] * n[1]; i++)
        m_T[i] *= n[0] * n[1];
}



void eval_T(complex<double> *T, complex<double> *m_T, array<size_t, 3> n, double q_p, double t)
{   using namespace std::complex_literals;
    complex<double> *m_T_0 = m_T,
                    *m_T_1 = m_T + n[0] * n[1];
    for (size_t i = 0; i < n[0] * n[1]; i++)
        T[i] = .5 * (m_T_0[i] + m_T_1[i] * exp(2i * M_PI * t / q_p));
}
