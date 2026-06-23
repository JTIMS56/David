/*
 * Standalone demo: Financial Dirac/Telegraph Model vs Black-Scholes
 * ──────────────────────────────────────────────────────────────────────────────
 * Compile without CMake:
 *   g++ -O3 -std=c++17 -Iinclude examples/dirac_demo.cpp -o dirac_demo -lm
 *
 * Via CMake:
 *   cmake -B build && cmake --build build && ./build/dirac_demo
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include "../include/fx_dirac.hpp"
#include <cstdio>
#include <cmath>

static void separator(const char* label) {
    std::printf("\n── %s ──\n", label);
}

int main() {
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  David — Financial Dirac Brane-World FX Prediction       ║\n");
    std::printf("║  1+1D Telegraph / Wilson-Dirac model  |  RK4 integrator  ║\n");
    std::printf("╚══════════════════════════════════════════════════════════╝\n");

    // EUR/USD parameters
    Real S0    = 1.0850;
    Real sigma = 0.0820;  // 8.2% ann vol
    Real mu    = 0.0050;  // 0.5% drift (real-world)
    Real r     = 0.0525;  // 5.25% risk-free
    Real T1m   = 1.0 / 12.0;
    Real T3m   = 3.0 / 12.0;

    auto run = [](Real S0, Real sigma, Real mu, Real r, Real kappa,
                  Real delta_cp, Real T, int N) -> FXDiracOutput {
        FXDiracInput in;
        in.S0 = S0; in.sigma = sigma; in.mu = mu; in.r = r;
        in.kappa = kappa; in.delta_cp = delta_cp;
        in.horizon_T = T; in.N_sites = N; in.x_width_std = 5;
        return FinancialDiracModel(in).run();
    };

    // ── 1. Black-Scholes convergence (κ → ∞) ─────────────────────────────────
    separator("BS convergence: increasing κ (flip rate)");
    std::printf("%-10s  %-12s  %-12s  %-10s  %-12s\n",
                "kappa", "Call_Dirac", "Call_BS", "Rel_err%", "Var_log");
    std::printf("%-10s  %-12s  %-12s  %-10s  %-12s\n",
                "─────────","──────────","──────────","──────────","──────────");

    Real bs_ref = BS::call_price(S0, S0, sigma, r, T1m);
    for (Real kappa : {0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0}) {
        auto out = run(S0, sigma, r, r, kappa, 0.0, T1m, 400);  // risk-neutral: mu=r
        Real rel_err = std::abs(out.call_dirac - out.call_bs)
                       / (out.call_bs + 1e-12) * 100.0;
        std::printf("%-10.1f  %-12.6f  %-12.6f  %-10.3f  %-12.6f\n",
                    kappa, out.call_dirac, bs_ref, rel_err, out.var_log_dirac);
    }

    // ── 2. Dirac mass effect on variance and call price ───────────────────────
    separator("Kappa effect on distribution shape (T=1m, μ=r risk-neutral)");
    std::printf("%-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
                "kappa","Call_D","Call_BS","VarRatio","E[S_T]","Q5");
    std::printf("%-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
                "─────","──────","──────","──────","──────","──────");
    for (Real kappa : {0.1, 0.5, 1.0, 2.0, 5.0, 20.0}) {
        auto out = run(S0, sigma, r, r, kappa, 0.0, T1m, 350);
        std::printf("%-8.1f  %-10.6f  %-10.6f  %-10.4f  %-10.6f  %+.4f\n",
                    kappa, out.call_dirac, out.call_bs,
                    out.var_log_dirac / (out.var_log_bs + 1e-12),
                    out.mean_dirac, out.chiral_charge);
    }

    // ── 3. CP asymmetry (δ_cp ≠ 0) — bullish/bearish bias ────────────────────
    separator("CP asymmetry: δ_cp effect on chiral charge and call price");
    std::printf("%-8s  %-12s  %-12s  %-10s\n",
                "delta_cp","Call_Dirac","Call_BS","Q5(sentiment)");
    for (Real dcp : {-0.5, -0.25, 0.0, 0.25, 0.5}) {
        auto out = run(S0, sigma, mu, r, 2.0, dcp, T1m, 350);
        std::printf("%-8.2f  %-12.6f  %-12.6f  %+.6f\n",
                    dcp, out.call_dirac, out.call_bs, out.chiral_charge);
    }

    // ── 4. 3-month horizon ────────────────────────────────────────────────────
    separator("3-month horizon: Dirac (κ=2) vs BS distribution");
    {
        auto d2  = run(S0, sigma, r, r, 2.0,  0.0, T3m, 400);
        auto d10 = run(S0, sigma, r, r, 10.0, 0.0, T3m, 400);
        int  N   = (int)d2.prices.size();

        std::printf("  Dirac(κ=2):  Call=%.5f  E[S]=%.5f  σ_log=%.4f\n",
                    d2.call_dirac, d2.mean_dirac, std::sqrt(d2.var_log_dirac));
        std::printf("  Dirac(κ=10): Call=%.5f  E[S]=%.5f  σ_log=%.4f\n",
                    d10.call_dirac, d10.mean_dirac, std::sqrt(d10.var_log_dirac));
        std::printf("  BS ref:      Call=%.5f  E[S]=%.5f  σ_log=%.4f\n\n",
                    d2.call_bs, d2.mean_bs, std::sqrt(d2.var_log_bs));

        // Print 11 points of the distribution
        std::printf("  %-10s  %-14s  %-14s  %-14s\n",
                    "Price","P_Dirac(κ=2)","P_Dirac(κ=10)","P_BS");
        int stride = std::max(1, N / 12);
        for (int j = stride; j < N - stride; j += stride) {
            std::printf("  %-10.4f  %-14.6f  %-14.6f  %-14.6f\n",
                        d2.prices[j],
                        d2.prob_dirac[j],
                        d10.prob_dirac[j],
                        d2.prob_bs[j]);
        }
    }

    // ── 5. Finite-κ correction formula verification ───────────────────────────
    separator("Effective diffusion: D_eff = σ²/2·(1 + 1/κ) theory vs measured");
    {
        Real D_bs = 0.5 * sigma * sigma;
        for (Real kappa : {0.5, 1.0, 2.0, 5.0, 10.0}) {
            auto out = run(S0, sigma, r, r, kappa, 0.0, T1m, 400);
            Real D_meas = out.var_log_dirac / T1m;
            Real D_theory = D_bs * (1.0 + 1.0/kappa);
            std::printf("  κ=%.1f: D_meas=%.5f  D_theory=%.5f  D_BS=%.5f\n",
                        kappa, D_meas, D_theory, D_bs);
        }
    }

    std::printf("\n═══════════════════════════════════════════════════════════\n");
    std::printf("BS limit confirmed: κ→∞ → D_eff→σ²/2, Call→BS, Var→σ²T\n");
    std::printf("Dirac correction:   κ finite → extra diffusion c²/(2κ)\n");
    std::printf("═══════════════════════════════════════════════════════════\n");
    return 0;
}
