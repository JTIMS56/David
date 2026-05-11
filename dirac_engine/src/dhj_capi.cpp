/*
 * DHJ C API — plain-C interface for Python ctypes
 */
#include "../include/dhj_model.hpp"
#include <cstring>
#include <cstdio>

extern "C" {

/*
 * dhj_predict — run the Dirac-Heston-Jump model.
 *
 * Heston params: kappa_H, theta, xi, rho, v0
 * Dirac params:  kappa0, kappa1, delta_cp
 * Jump params:   lambda, p_up, eta_plus, eta_minus
 *
 * scalars_out[13] = {mean_dhj, mean_bs, var_dhj, var_bs,
 *                    call_dhj, call_bs, chiral, avg_var,
 *                    n_steps, n_paths, mass_loss_fraction,
 *                    min_density, negative_count}
 *
 * Returns: actual points filled, or -1 on error.
 */
int dhj_predict(
    // Market — Garman-Kohlhagen: r_d = domestic, r_f = foreign (0 for equity)
    double S0, double r_d, double r_f, double horizon_T,
    // Heston
    double kappa_H, double theta, double xi, double rho, double v0,
    // Dirac
    double kappa0, double kappa1, double delta_cp,
    // Jumps
    double lambda, double p_up, double eta_plus, double eta_minus,
    // Lattice + MC
    int n_sites, int n_paths, unsigned seed,
    // Output arrays
    double* prices_out, double* prob_dhj_out, double* prob_bs_out,
    int n_out,
    double* scalars_out   // length 13
) {
    try {
        DHJInput in;
        in.S0        = S0;
        in.r         = r_d;
        in.r_f       = r_f;
        in.horizon_T = horizon_T;

        in.heston.kappa_H = kappa_H;
        in.heston.theta   = theta;
        in.heston.xi      = xi;
        in.heston.rho     = rho;
        in.heston.v0      = v0;

        in.kappa0   = kappa0;
        in.kappa1   = kappa1;
        in.delta_cp = delta_cp;

        in.jumps.lambda    = lambda;
        in.jumps.p_up      = p_up;
        in.jumps.eta_plus  = eta_plus;
        in.jumps.eta_minus = eta_minus;

        in.N_sites     = n_sites > 0 ? n_sites : 400;
        in.x_width_std = 5;
        in.N_paths     = n_paths > 0 ? n_paths : 300;
        in.seed        = seed;

        DHJOutput out = DHJModel(in).run();

        int n_fill = std::min(n_out, (int)out.prices.size());
        for (int i = 0; i < n_fill; ++i) {
            if (prices_out)    prices_out[i]    = out.prices[i];
            if (prob_dhj_out)  prob_dhj_out[i]  = out.prob_dhj[i];
            if (prob_bs_out)   prob_bs_out[i]   = out.prob_bs[i];
        }
        if (scalars_out) {
            scalars_out[0]  = out.mean_dhj;
            scalars_out[1]  = out.mean_bs;
            scalars_out[2]  = out.var_log_dhj;
            scalars_out[3]  = out.var_log_bs;
            scalars_out[4]  = out.call_dhj;
            scalars_out[5]  = out.call_bs;
            scalars_out[6]  = out.chiral_charge;
            scalars_out[7]  = out.avg_variance;
            scalars_out[8]  = (double)out.n_steps;
            scalars_out[9]  = (double)out.n_paths;
            scalars_out[10] = out.mass_loss_fraction;
            scalars_out[11] = out.min_density;
            scalars_out[12] = (double)out.negative_count;
        }
        return n_fill;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[dhj_predict] %s\n", e.what());
        return -1;
    }
}

void dhj_version(char* buf, int len) {
    std::snprintf(buf, len,
        "DiracFX-DHJ 1.0  |  Dirac-Heston-Jump  |  MC-PDE hybrid  |  RK4+Milstein");
}

} // extern "C"
