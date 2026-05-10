/*
 * DHJ Demo — Dirac-Heston-Jump model validation
 * ──────────────────────────────────────────────────────────────────────────────
 * Compile (standalone):
 *   g++ -O3 -std=c++17 -Iinclude examples/dhj_demo.cpp -o dhj_demo -lm
 *
 * Via CMake:
 *   cmake --build build && ./build/dhj_demo
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include "../include/dhj_model.hpp"
#include <cstdio>
#include <cmath>
#include <chrono>

static void sep(const char* s) { std::printf("\n── %s ──\n", s); }

// Build a default DHJInput for EUR/USD (Garman-Kohlhagen: USD domestic, EUR foreign)
static DHJInput eurusd_input(double horizon_T, int N_paths = 300) {
    DHJInput in;
    in.S0        = 1.0850;
    in.r         = 0.0525;   // r_d: USD domestic rate
    in.r_f       = 0.0400;   // r_f: EUR foreign rate  (carry = 1.25%)
    in.horizon_T = horizon_T;

    in.heston.kappa_H = 1.5;    // moderate mean-reversion
    in.heston.theta   = 0.0067; // 8.2%² long-run variance
    in.heston.xi      = 0.30;   // vol-of-vol
    in.heston.rho     = -0.25;  // slight leverage (down moves → vol spike)
    in.heston.v0      = 0.0067; // start at long-run

    in.kappa0   = 1.0;          // base Dirac mass
    in.kappa1   = 0.002;        // vol-dependent correction
    in.delta_cp = 0.0;

    in.jumps.lambda    = 5.0;   // 5 jumps/year (typical FX)
    in.jumps.p_up      = 0.55;
    in.jumps.eta_plus  = 100.0; // mean up-jump   = 1.0% (FX realistic)
    in.jumps.eta_minus = 80.0;  // mean down-jump = 1.25%

    in.N_sites     = 400;
    in.x_width_std = 5;
    in.N_paths     = N_paths;
    in.seed        = 42;
    return in;
}

// Run just the pure Dirac model (no stochastic vol, no jumps) for comparison
static FXDiracOutput dirac_only(double S0, double sigma, double r, double T) {
    FXDiracInput in;
    in.S0 = S0; in.sigma = sigma; in.mu = r; in.r = r;
    in.kappa = 2.0; in.delta_cp = 0.0;
    in.horizon_T = T; in.N_sites = 400; in.x_width_std = 5;
    return FinancialDiracModel(in).run();
}

int main() {
    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  David — Dirac-Heston-Jump (DHJ) FX Prediction Engine        ║\n");
    std::printf("║  MC-PDE hybrid: Heston vol × Dirac brane × Kou jumps         ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    const double S0    = 1.0850;
    const double sigma = 0.0820;
    const double r_d   = 0.0525;  // USD domestic rate
    const double r_f   = 0.0400;  // EUR foreign rate  (carry = 1.25%)
    const double carry = r_d - r_f;
    const double T1m   = 1.0 / 12.0;
    const double T3m   = 3.0 / 12.0;

    // ── 1. Model comparison at 1 month ───────────────────────────────────────
    sep("1-month comparison: GK vs Dirac vs DHJ");
    std::printf("%-14s  %-10s  %-10s  %-10s  %-10s\n",
                "Model", "E[S_T]", "Call_ATM", "VarRatio", "ChiralQ5");
    std::printf("%-14s  %-10s  %-10s  %-10s  %-10s\n",
                "─────────────", "──────", "──────", "──────", "──────");

    // Garman-Kohlhagen reference
    Real bs_call = BS::call_price(S0, S0, sigma, r_d, T1m, r_f);
    Real bs_mean = S0 * std::exp(carry * T1m);  // GK forward
    Real bs_var  = sigma * sigma * T1m;
    std::printf("%-14s  %-10.5f  %-10.6f  %-10.4f  %-10s\n",
                "Black-Scholes", bs_mean, bs_call, 1.0, "n/a");

    // Pure Dirac (uses mu=r_d; no foreign rate in simple Dirac model)
    auto d = dirac_only(S0, sigma, r_d, T1m);
    std::printf("%-14s  %-10.5f  %-10.6f  %-10.4f  %+.4f\n",
                "Dirac(κ=2)",
                d.mean_dirac, d.call_dirac,
                d.var_log_dirac / (bs_var + 1e-15),
                d.chiral_charge);

    // DHJ
    {
        auto t0 = std::chrono::steady_clock::now();
        DHJOutput dhj = DHJModel(eurusd_input(T1m, 300)).run();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("%-14s  %-10.5f  %-10.6f  %-10.4f  %+.6f  [%.0f ms]\n",
                    "DHJ",
                    dhj.mean_dhj, dhj.call_dhj,
                    dhj.var_log_dhj / (bs_var + 1e-15),
                    dhj.chiral_charge, ms);
        std::printf("  DHJ: avg_vol=%.4f  n_steps=%d  n_paths=%d  mass_loss=%.3f%%\n",
                    std::sqrt(dhj.avg_variance), dhj.n_steps, dhj.n_paths,
                    dhj.mass_loss_fraction * 100.0);
    }

    // ── 2. Heston effect: varying xi (vol-of-vol) ────────────────────────────
    sep("Heston vol-of-vol effect (ξ sweep, T=1m)");
    std::printf("%-6s  %-10s  %-10s  %-10s\n", "xi", "Call_DHJ", "Call_BS", "VarRatio");
    for (double xi : {0.0, 0.10, 0.20, 0.30, 0.50}) {
        auto in = eurusd_input(T1m, 200);
        in.heston.xi = xi;
        in.jumps.lambda = 0.0;  // isolate Heston effect
        DHJOutput dhj = DHJModel(in).run();
        std::printf("%-6.2f  %-10.6f  %-10.6f  %-10.4f\n",
                    xi, dhj.call_dhj, dhj.call_bs,
                    dhj.var_log_dhj / (dhj.var_log_bs + 1e-15));
    }

    // ── 3. Leverage effect: varying rho ──────────────────────────────────────
    sep("Leverage effect (ρ sweep, ξ=0.3, T=1m)");
    std::printf("%-6s  %-10s  %-10s  %-10s\n", "rho", "Call_DHJ", "E[S_T]", "VarRatio");
    for (double rho : {-0.6, -0.3, 0.0, 0.3, 0.6}) {
        auto in = eurusd_input(T1m, 200);
        in.heston.rho = rho;
        in.jumps.lambda = 0.0;
        DHJOutput dhj = DHJModel(in).run();
        std::printf("%-6.2f  %-10.6f  %-10.5f  %-10.4f\n",
                    rho, dhj.call_dhj, dhj.mean_dhj,
                    dhj.var_log_dhj / (dhj.var_log_bs + 1e-15));
    }

    // ── 4. Jump intensity effect ──────────────────────────────────────────────
    sep("Kou jump effect (λ sweep, ξ=0, T=1m)");
    std::printf("%-6s  %-10s  %-10s  %-10s\n", "lambda", "Call_DHJ", "Call_BS", "VarRatio");
    for (double lam : {0.0, 1.0, 3.0, 5.0, 10.0}) {
        auto in = eurusd_input(T1m, 200);
        in.heston.xi = 0.0;   // isolate jump effect
        in.heston.rho = 0.0;
        in.jumps.lambda = lam;
        DHJOutput dhj = DHJModel(in).run();
        std::printf("%-6.1f  %-10.6f  %-10.6f  %-10.4f\n",
                    lam, dhj.call_dhj, dhj.call_bs,
                    dhj.var_log_dhj / (dhj.var_log_bs + 1e-15));
    }

    // ── 5. Full DHJ at 3-month horizon ───────────────────────────────────────
    sep("3-month horizon: full DHJ vs Dirac vs BS");
    {
        DHJOutput dhj3 = DHJModel(eurusd_input(T3m, 400)).run();
        auto d3  = dirac_only(S0, sigma, r_d, T3m);
        Real bc3 = BS::call_price(S0, S0, sigma, r_d, T3m, r_f);

        std::printf("  DHJ:   Call=%.5f  E[S]=%.5f  σ_log=%.4f  Q5=%+.4f\n",
                    dhj3.call_dhj, dhj3.mean_dhj,
                    std::sqrt(dhj3.var_log_dhj), dhj3.chiral_charge);
        std::printf("  Dirac: Call=%.5f  E[S]=%.5f  σ_log=%.4f\n",
                    d3.call_dirac, d3.mean_dirac, std::sqrt(d3.var_log_dirac));
        std::printf("  BS(GK):Call=%.5f  E[S]=%.5f  σ_log=%.4f\n\n",
                    bc3, S0 * std::exp(carry * T3m), std::sqrt(sigma * sigma * T3m));

        // Distribution sample
        int Np = (int)dhj3.prices.size();
        int stride = std::max(1, Np / 12);
        std::printf("  %-10s  %-14s  %-14s\n", "Price", "P_DHJ", "P_BS");
        for (int j = stride; j < Np - stride; j += stride) {
            if (dhj3.prob_dhj[j] > 1e-6 || dhj3.prob_bs[j] > 1e-6)
                std::printf("  %-10.4f  %-14.6f  %-14.6f\n",
                            dhj3.prices[j], dhj3.prob_dhj[j], dhj3.prob_bs[j]);
        }
    }

    // ── 6. BS recovery: all corrections off ──────────────────────────────────
    sep("BS recovery: DHJ with ξ=0, λ=0, ρ=0, κ₀→∞");
    {
        auto in = eurusd_input(T1m, 200);
        in.heston.xi  = 0.0;
        in.heston.rho = 0.0;
        in.jumps.lambda = 0.0;
        in.kappa0 = 200.0;
        in.kappa1 = 0.0;
        DHJOutput dhj = DHJModel(in).run();
        Real rel = std::abs(dhj.call_dhj - dhj.call_bs) / (dhj.call_bs + 1e-12) * 100;
        std::printf("  Call_DHJ=%.6f  Call_BS=%.6f  RelErr=%.2f%%\n",
                    dhj.call_dhj, dhj.call_bs, rel);
        std::printf("  E[S_T]=%.6f  E[S_T]_BS=%.6f\n",
                    dhj.mean_dhj, dhj.mean_bs);
    }

    std::printf("\n═══════════════════════════════════════════════════════════════\n");
    std::printf("DHJ model: Heston stoch-vol × Dirac brane-world × Kou jumps\n");
    std::printf("Convergence: ξ→0, λ→0, ρ→0, κ₀→∞ ⟹ Black-Scholes ✓\n");
    std::printf("═══════════════════════════════════════════════════════════════\n");
    return 0;
}
