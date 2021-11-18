#pragma once
#include <array>
#include <complex>
#include <fftw3.h>

void linear_fill(double *const data, const std::size_t n, const double begin, const double step)
{   for (std::size_t i = 0; i < n; i++)
        data[i] = begin + step * i;
}



void eval_intensity_freq_space(std::complex<double> *const q_ft, const std::array<std::size_t, 3> n,
                               const double *const k_x, const double *const k_y, const double q_P,
                               const std::array<double, 2> q_center, const double q_sd)
{   using namespace std::complex_literals;
    for (std::size_t j = 0; j < n[1]; j++)
        for (std::size_t i = 0; i < n[0]; i++)
            q_ft[i + j * n[0]] = q_P * exp(-2 * M_PI * M_PI * q_sd * q_sd * (k_x[i] * k_x[i] + k_y[j] * k_y[j]))
                                     * exp(-2i * M_PI * (q_center[0] * k_x[i] + q_center[1] * k_y[j]));
}



void eval_sigma_0(std::complex<double> *const s, const std::array<std::size_t, 3> n,
                  const double *const k_x, const double *const k_y, const std::array<double, 3> a)
{   for (std::size_t j = 0; j < n[1]; j++)
        for (std::size_t i = 0; i < n[0]; i++)
            s[i + j * n[0]] = 2 * M_PI * sqrt((a[0] * (k_x[i] * k_x[i])
                                             + a[1] * (k_y[j] * k_y[j])) / a[2]);
}

void eval_sigma_1(std::complex<double> *const s, const std::array<std::size_t, 3> n,
                  const double *const k_x, const double *const k_y, const std::array<double, 3> a, const double q_p)
{   using namespace std::complex_literals;
    for (std::size_t j = 0; j < n[1]; j++)
        for (std::size_t i = 0; i < n[0]; i++)
            s[i + j * n[0]] = sqrt(2 * M_PI) * sqrt((1i / q_p + 2 * M_PI * a[0] * k_x[i] * k_x[i]
                                                              + 2 * M_PI * a[1] * k_y[j] * k_y[j]) / a[2]);
}



void eval_time_mutual_T_freq_space(std::complex<double> *const m_T_dft, const std::array<std::size_t, 3> n,
                                   const std::complex<double> *const q_dft, const std::complex<double> *const s,
                                   const double b_z, const double z, const double d)
{   for (std::size_t i = 0; i < n[0] * n[1]; i++)
    {   std::complex<double> a = q_dft[i] / (b_z * s[i]),
                        b = z * s[i],
                        c  = 2 * d * s[i];
        //  the below approximates a * (exp(b) / (1. - exp(c)) - exp(-b) / (1. - exp(-c))), which is too unstable due to large c
        m_T_dft[i] = a * (-exp(b - c) - exp(-b) / (1. - exp(-c)));
    }
}

void eval_time_mutual_T_real_space(std::complex<double> *m_T, std::complex<double> *m_T_dft,
                                   const std::array<std::size_t, 3> n)
{   fftw_plan plan = fftw_plan_dft_2d(n[1], n[0],
                                      reinterpret_cast<fftw_complex *>(m_T_dft), reinterpret_cast<fftw_complex *>(m_T),
                                      FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    for (std::size_t i = 0; i < n[0] * n[1]; i++)
        m_T[i] *= n[0] * n[1];
}



void eval_T(std::complex<double> *const T, const std::complex<double> *const m_T,
            const std::array<std::size_t, 3> n, const double q_p, const double t)
{
    using namespace std::complex_literals;
    for (std::size_t i = 0; i < n[0] * n[1]; i++)
        T[i] = .5 * (m_T[i] + (m_T + n[0] * n[1])[i] * exp(2i * M_PI * t / q_p));
}
