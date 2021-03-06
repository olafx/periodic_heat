#include "solver_sine.hh"
#include "storage.hh"

int main()
{
    //  independent settings

    /* real space dims */                 std::array<std::size_t, 3> n {768, 128, 128};
    /* heat diffusivities (x, y, z) */    std::array<double, 3> a      {3e-7, 1e-7, 1e-7};
    /* real space range (x, y, z) */      std::array<double, 6> range  {0, .1, 0, .1, 0, 4e-3};
    /* heat conductivity along z */       double b_z  = .1;
    /* laser power */                     double q_P  = .5;
    /* period of laser intensity */       double q_p  = 10;
    /* real space sd of Gaussian laser */ double q_sd = 1e-3;
    /* evaluation time */                 double t    = 7;

    //  dependent settings

    std::array<double, 2> q_center {.5 * (range[0] + range[1]), .5 * (range[2] + range[3])};



    //  space and frequency space coordinates

    double *z   = new double[n[2]];
    double *k_x = new double[n[0]];
    double *k_y = new double[n[1]];
    linear_fill(z,   n[2], range[4],                                     (range[5] - range[4]) / n[2]);
    linear_fill(k_x, n[0], 0,                                        1 / (range[1] - range[0]));
    linear_fill(k_y, n[1], -.5 * (n[1] - 1) / (range[3] - range[2]), 1 / (range[3] - range[2]));



    //  frequency space Gaussian intensity

    auto *q_ft = new std::complex<double>[n[0] * n[1]];
    eval_intensity_freq_space(q_ft, n, k_x, k_y, q_P, q_center, q_sd);



    //  sigma for frequencies 0 and 1 / q_p

    //  s storage format:
    //      freq 0       pos k_x[0] k_y[0],        ..., freq 0       pos k_x[n[0] - 1] k_y[0],
    //      ...,
    //      freq 0       pos k_x[0] k_y[n[1] - 1], ..., freq 0       pos k_x[n[0] - 1] k_y[n[1] - 1],
    //      freq 1 / q_p pos k_x[0] k_y[0],        ..., freq 1 / q_p pos k_x[n[0] - 1] k_y[0],
    //      ...,
    //      freq 1 / q_p pos k_x[0] k_y[n[1] - 1], ..., freq 1 / q_p pos k_x[n[0] - 1] k_y[n[1] - 1]
    auto *s = new std::complex<double>[n[0] * n[1] * 2];
    eval_sigma_0(s              , n, k_x, k_y, a);
    eval_sigma_1(s + n[0] * n[1], n, k_x, k_y, a, q_p);
    delete[] k_x;
    delete[] k_y;



    //  z loop

    /*  m_T_dft storage format:
        freq 0       pos k_x[0] k_y[0],        ..., freq 0       pos k_x[n[0] - 1] k_y[0],
        ...,
        freq 0       pos k_x[0] k_y[n[1] - 1], ..., freq 0       pos k_x[n[0] - 1] k_y[n[1] - 1],
        freq 1 / q_p pos k_x[0] k_y[0],        ..., freq 1 / q_p pos k_x[n[0] - 1] k_y[0],
        ...,
        freq 1 / q_p pos k_x[0] k_y[n[1] - 1], ..., freq 1 / q_p pos k_x[n[0] - 1] k_y[n[1] - 1]

        m_T storage format:
        freq 0       pos x[0] y[0],        ..., freq 0       pos x[n[0] - 1] y[0],
        ...,
        freq 0       pos x[0] y[n[1] - 1], ..., freq 0       pos x[n[0] - 1] y[n[1] - 1],
        freq 1 / q_p pos x[0] y[0],        ..., freq 1 / q_p pos x[n[0] - 1] y[0],
        ...,
        freq 1 / q_p pos x[0] y[n[1] - 1], ..., freq 1 / q_p pos x[n[0] - 1] y[n[1] - 1]

        T storage format:
        pos x[0] y[0]        z[0],        ..., pos x[n[0] - 1] y[0]        z[0],
        ...,
        pos x[0] y[n[1] - 1] z[0],        ..., pos x[n[0] - 1] y[n[1] - 1] z[0],
        ...,
        pos x[0] y[0]        z[n[2] - 1], ..., pos x[n[0] - 1] y[0]        z[n[2] - 1],
        ...,
        pos x[0] y[n[1] - 1] z[n[2] - 1], ..., pos x[n[0] - 1] y[n[1] - 1] z[n[2] - 1],
    */
    auto *m_T_dft = new std::complex<double>[n[0] * n[1] * 2];
    auto *m_T     = new std::complex<double>[n[0] * n[1] * 2];
    auto *T       = new std::complex<double>[n[0] * n[1] * n[2]];

    for (std::size_t i = 0; i < n[2]; i++)
    {
        //  time mutual T components for frequencies 0 and 1 / q_p in frequency space
        eval_time_mutual_T_freq_space(m_T_dft,               n, q_ft, s,               b_z, z[i], z[n[2] - 1]);
        eval_time_mutual_T_freq_space(m_T_dft + n[0] * n[1], n, q_ft, s + n[0] * n[1], b_z, z[i], z[n[2] - 1]);

        //  time mutual T components for frequencies 0 and 1 / q_p in real space
        eval_time_mutual_T_real_space(m_T,               m_T_dft,               n);
        eval_time_mutual_T_real_space(m_T + n[0] * n[1], m_T_dft + n[0] * n[1], n);

        //  T in real space
        eval_T((T + i * n[0] * n[1]), m_T, n, q_p, t);
    }



    delete[] z;
    delete[] q_ft;
    delete[] s;
    delete[] m_T_dft;
    delete[] m_T;



    //  storage

    store(static_cast<void *>(T), n, "0.vti");

    delete[] T;
}
