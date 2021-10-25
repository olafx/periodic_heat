#pragma once
#include <cstddef>
#include <array>
#include <cmath>
#include <complex>
#include <fftw3.h>

void linear_fill(double *data, size_t n, double begin, double step)
{   for (size_t i = 0; i < n; i++)
    {   data[i] = begin + step * i;
    }
}

void gaussian_intensity(double *q, std::array<size_t, 2> n, double *x, double *y,
                        double q_P, std::array<double, 2> q_center, double q_sd)
{   double a = .5 * M_1_PI * q_P / (q_sd * q_sd);
    for (size_t j = 0; j < n[1]; j++)
        for (size_t i = 0; i < n[0]; i++)
        {   double b1 = (x[i] - q_center[0]) / q_sd;
            double b2 = (y[j] - q_center[1]) / q_sd;
            q[i + j * n[0]] = a * exp(-.5 * (b1 * b1 + b2 * b2));
        }
}

void s_0(std::complex<double> *s, std::array<size_t, 2> n, double *k_x, double *k_y, std::array<double, 3> a)
{
}

void s_1(std::complex<double> *s, std::array<size_t, 2> n, double *k_x, double *k_y, std::array<double, 3> a, double q_p)
{
}
