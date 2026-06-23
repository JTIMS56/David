/*
 * C API — plain-C interface for Python ctypes (no name mangling)
 */
#include "../include/fx_dirac.hpp"
#include <cstring>
#include <cstdio>

extern "C" {

/*
 * dirac_predict — run the Dirac FX model, fill output arrays.
 *
 * Outputs:
 *   prices_out, prob_dirac_out, prob_bs_out  — length n_out each
 *   scalars_out  [8] = {mean_d, mean_bs, var_d, var_bs,
 *                        call_d, call_bs, chiral, n_steps}
 * Returns: actual points filled, or -1 on error.
 */
int dirac_predict(
    double S0, double sigma, double mu, double r,
    double kappa, double delta_cp, double horizon_T,
    int n_sites,
    double* prices_out, double* prob_dirac_out, double* prob_bs_out,
    int n_out,
    double* scalars_out
) {
    try {
        FXDiracInput in;
        in.S0 = S0; in.sigma = sigma; in.mu = mu; in.r = r;
        in.kappa = kappa; in.delta_cp = delta_cp;
        in.horizon_T = horizon_T;
        in.N_sites = (n_sites > 0) ? n_sites : 400;
        in.x_width_std = 5;

        FXDiracOutput out = FinancialDiracModel(in).run();

        int n_fill = std::min(n_out, (int)out.prices.size());
        for (int i = 0; i < n_fill; ++i) {
            if (prices_out)     prices_out[i]     = out.prices[i];
            if (prob_dirac_out) prob_dirac_out[i]  = out.prob_dirac[i];
            if (prob_bs_out)    prob_bs_out[i]     = out.prob_bs[i];
        }
        if (scalars_out) {
            scalars_out[0] = out.mean_dirac;
            scalars_out[1] = out.mean_bs;
            scalars_out[2] = out.var_log_dirac;
            scalars_out[3] = out.var_log_bs;
            scalars_out[4] = out.call_dirac;
            scalars_out[5] = out.call_bs;
            scalars_out[6] = out.chiral_charge;
            scalars_out[7] = (double)out.n_steps;
        }
        return n_fill;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[dirac_predict] %s\n", e.what());
        return -1;
    }
}

/*
 * dirac_bs_convergence_test — verify κ→∞ gives BS within tol.
 * Returns 1 if pass, 0 if fail.
 */
int dirac_bs_convergence_test(
    double S0, double sigma, double r, double T,
    double tol,
    double* call_dirac_out, double* call_bs_out
) {
    try {
        FXDiracInput in;
        in.S0 = S0; in.sigma = sigma; in.mu = r; in.r = r;
        in.kappa = 200.0;          // heavily overdamped → BS
        in.delta_cp = 0.0;
        in.horizon_T = T;
        in.N_sites = 500; in.x_width_std = 6;

        FXDiracOutput out = FinancialDiracModel(in).run();
        if (call_dirac_out) *call_dirac_out = out.call_dirac;
        if (call_bs_out)    *call_bs_out    = out.call_bs;

        double rel = std::abs(out.call_dirac - out.call_bs) / (out.call_bs + 1e-12);
        return (rel < tol) ? 1 : 0;
    } catch (...) { return 0; }
}

void dirac_version(char* buf, int len) {
    std::snprintf(buf, len,
        "DiracFX 1.0  |  1+1D Telegraph/Wilson-Dirac  |  RK4  |  BS-exact limit");
}

} // extern "C"
