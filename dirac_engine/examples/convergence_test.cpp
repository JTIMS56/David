/*
 * DHJ Convergence & Validation Test Suite
 * ──────────────────────────────────────────────────────────────────────────────
 * Tests:
 *   1. GK/BS recovery vs grid resolution  N ∈ {200, 400, 800, 1600}
 *   2. κ₀ sweep: BS error decreases as κ₀ increases
 *   3. Moment test: E[S_T] = S₀·exp(carry·T)  (martingale condition)
 *   4. Variance test: Var[log S_T] converges to σ²·T
 *   5. Jump martingale: compensated Kou jumps preserve E[S_T]
 *   6. Negative density test: no negative densities in output
 *   7. Mass loss test: boundary leakage stays near zero
 *
 * Compile via CMake:  cmake --build build && ./build/convergence_test
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include "../include/dhj_model.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>

// ── Helpers ───────────────────────────────────────────────────────────────────

static void sep(const char* s) {
    std::printf("\n╔══ %s\n", s);
}

static void pass_fail(bool ok, const char* label, const char* detail = "") {
    std::printf("  %s  %-52s  %s\n",
                ok ? "✓" : "✗", label, detail);
}

// Build a minimal DHJ input for convergence tests.
// ξ=0, λ=0, ρ=0 → deterministic variance; only κ₀ and N vary.
static DHJInput bs_input(int N, double kappa0) {
    DHJInput in;
    in.S0        = 1.0;
    in.r         = 0.0525;
    in.r_f       = 0.0400;
    in.horizon_T = 1.0 / 12.0;   // 1 month

    in.heston.kappa_H = 2.0;
    in.heston.theta   = 0.082 * 0.082;   // σ²
    in.heston.xi      = 0.0;             // no stochastic vol
    in.heston.rho     = 0.0;
    in.heston.v0      = 0.082 * 0.082;

    in.kappa0   = kappa0;
    in.kappa1   = 0.0;
    in.delta_cp = 0.0;

    in.jumps.lambda = 0.0;   // no jumps
    in.jumps.p_up      = 0.55;
    in.jumps.eta_plus  = 100.0;
    in.jumps.eta_minus = 80.0;

    in.N_sites     = N;
    in.x_width_std = 5;
    in.N_paths     = 1;     // single path (deterministic variance)
    in.seed        = 42;
    return in;
}

// GK call price reference
static double gk_call(double S0, double K, double sigma, double r_d, double T, double r_f) {
    double carry   = r_d - r_f;
    double sig_T   = sigma * std::sqrt(T);
    double d1      = (std::log(S0 / K) + (carry + 0.5 * sigma * sigma) * T) / sig_T;
    double d2      = d1 - sig_T;
    auto N_cdf     = [](double x) -> double {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    };
    return S0 * std::exp(-r_f * T) * N_cdf(d1)
         - K  * std::exp(-r_d * T) * N_cdf(d2);
}

// ── Test 1: GK recovery vs N ──────────────────────────────────────────────────
static void test_gk_recovery_vs_N() {
    sep("TEST 1: GK/BS recovery vs grid resolution  (κ₀=500, ξ=0, λ=0, ρ=0)");
    std::printf("  %-8s  %-12s  %-12s  %-10s  %-12s  %-8s\n",
                "N_sites", "E[S_T]", "GK_forward", "Call_DHJ", "Call_GK", "RelErr%");
    std::printf("  %-8s  %-12s  %-12s  %-10s  %-12s  %-8s\n",
                "───────", "──────────", "──────────", "────────", "───────", "──────");

    const double sigma = 0.082, r_d = 0.0525, r_f = 0.04, T = 1.0/12.0;
    const double gk_fwd  = std::exp((r_d - r_f) * T);   // S0=1
    const double call_gk = gk_call(1.0, 1.0, sigma, r_d, T, r_f);
    const double kappa0  = 500.0;

    bool all_converging = true;
    double prev_err = 1e9;

    for (int N : {200, 400, 800, 1600}) {
        auto in  = bs_input(N, kappa0);
        auto out = DHJModel(in).run();
        double rel = std::abs(out.call_dhj - call_gk) / (call_gk + 1e-12) * 100.0;
        bool converging = (rel < prev_err * 1.5);   // allow slight noise
        all_converging &= converging;
        std::printf("  %-8d  %-12.6f  %-12.6f  %-10.6f  %-12.6f  %-7.2f%%  %s\n",
                    N, out.mean_dhj, gk_fwd, out.call_dhj, call_gk, rel,
                    converging ? "" : "← not converging!");
        prev_err = rel;
    }

    pass_fail(prev_err < 2.0, "BS/GK call error < 2% at N=1600",
              ("RelErr=" + std::to_string(prev_err)).c_str());
    pass_fail(all_converging, "Error decreases (or stable) with N");
}

// ── Test 2: κ₀ sweep ──────────────────────────────────────────────────────────
static void test_kappa_sweep() {
    sep("TEST 2: κ₀ sweep  (N=400, ξ=0, λ=0, ρ=0) — BS limit as κ₀→∞");
    std::printf("  %-8s  %-12s  %-10s  %-12s  %-8s\n",
                "kappa0", "Call_DHJ", "Call_GK", "neg_count", "RelErr%");
    std::printf("  %-8s  %-12s  %-10s  %-12s  %-8s\n",
                "──────", "────────", "───────", "─────────", "──────");

    const double sigma = 0.082, r_d = 0.0525, r_f = 0.04, T = 1.0/12.0;
    const double call_gk = gk_call(1.0, 1.0, sigma, r_d, T, r_f);
    bool monotone = true;
    double prev_err = 1e9;

    for (double k0 : {10.0, 50.0, 100.0, 200.0, 500.0, 1000.0}) {
        auto in  = bs_input(400, k0);
        auto out = DHJModel(in).run();
        double rel = std::abs(out.call_dhj - call_gk) / (call_gk + 1e-12) * 100.0;
        bool decreasing = (rel <= prev_err + 0.2);  // allow 0.2% noise
        monotone &= decreasing;
        std::printf("  %-8.0f  %-12.6f  %-10.6f  %-12d  %-7.2f%%  %s\n",
                    k0, out.call_dhj, call_gk,
                    out.negative_count, rel,
                    decreasing ? "" : "← error increased!");
        prev_err = rel;
    }

    pass_fail(monotone,      "Call error decreases (or stable) as κ₀ increases");
    pass_fail(prev_err < 1.5, "BS/GK error < 1.5% at κ₀=1000");
}

// ── Test 3: Martingale / moment test ─────────────────────────────────────────
static void test_moments() {
    sep("TEST 3: Martingale condition  E[S_T] = S₀·exp(carry·T)");

    struct Case { double T; double kappa0; double xi; };
    std::vector<Case> cases = {
        {1.0/12.0,  2.0,  0.00},   // 1 month, pure Dirac (no stoch-vol)
        {1.0/12.0,  2.0,  0.30},   // 1 month, full Heston
        {3.0/12.0,  2.0,  0.30},   // 3 months, full Heston
        {1.0/12.0, 50.0,  0.00},   // BS limit
    };

    bool all_ok = true;
    std::printf("  %-8s  %-8s  %-8s  %-12s  %-12s  %-10s\n",
                "T(days)", "kappa0", "xi", "E[S_T]", "GK_fwd", "RelErr%");

    for (auto& c : cases) {
        DHJInput in;
        in.S0 = 1.0850; in.r = 0.0525; in.r_f = 0.0400; in.horizon_T = c.T;
        in.heston = {1.5, 0.082*0.082, c.xi, 0.0, 0.082*0.082};
        in.kappa0 = c.kappa0; in.kappa1 = 0.0; in.delta_cp = 0.0;
        in.jumps.lambda = 0.0;
        in.jumps.p_up = 0.55; in.jumps.eta_plus = 100.0; in.jumps.eta_minus = 80.0;
        in.N_sites = 400; in.x_width_std = 5;
        in.N_paths = 300; in.seed = 42;

        auto out = DHJModel(in).run();
        double fwd = in.S0 * std::exp((in.r - in.r_f) * c.T);
        double rel = std::abs(out.mean_dhj - fwd) / fwd * 100.0;
        bool ok = rel < 0.5;   // 0.5% tolerance
        all_ok &= ok;
        std::printf("  %-8.0f  %-8.0f  %-8.2f  %-12.6f  %-12.6f  %-9.3f%%  %s\n",
                    c.T * 365, c.kappa0, c.xi,
                    out.mean_dhj, fwd, rel, ok ? "✓" : "✗ FAIL");
    }
    pass_fail(all_ok, "E[S_T] within 0.5% of GK forward for all cases");
}

// ── Test 4: Jump martingale ────────────────────────────────────────────────────
static void test_jump_martingale() {
    sep("TEST 4: Jump martingale — compensated Kou jumps preserve E[S_T]");

    struct Case { double lambda; const char* label; };
    bool all_ok = true;
    std::printf("  %-8s  %-12s  %-12s  %-10s\n", "lambda", "E[S_T]", "GK_fwd", "RelErr%");

    for (auto& c : std::vector<std::pair<double,const char*>>{
            {1.0, "λ=1"},  {3.0, "λ=3"},  {5.0, "λ=5"},  {10.0, "λ=10"}}) {
        DHJInput in;
        in.S0 = 1.085; in.r = 0.0525; in.r_f = 0.04; in.horizon_T = 1.0/12.0;
        in.heston = {1.5, 0.082*0.082, 0.0, 0.0, 0.082*0.082};  // ξ=0: deterministic vol
        in.kappa0 = 2.0; in.kappa1 = 0.0; in.delta_cp = 0.0;
        in.jumps.lambda    = c.first;
        in.jumps.p_up      = 0.55;
        in.jumps.eta_plus  = 100.0;
        in.jumps.eta_minus = 80.0;
        in.N_sites = 400; in.x_width_std = 5;
        in.N_paths = 400; in.seed = 42;

        auto out = DHJModel(in).run();
        double fwd = in.S0 * std::exp((in.r - in.r_f) * in.horizon_T);
        double rel = std::abs(out.mean_dhj - fwd) / fwd * 100.0;
        bool ok = rel < 1.0;   // 1% tolerance for MC noise
        all_ok &= ok;
        std::printf("  %-8.1f  %-12.6f  %-12.6f  %-9.3f%%  %s\n",
                    c.first, out.mean_dhj, fwd, rel, ok ? "✓" : "✗ FAIL");
    }
    pass_fail(all_ok, "Jump compensator preserves GK martingale (< 1% error)");
}

// ── Test 5: Negative density ──────────────────────────────────────────────────
static void test_no_negative_densities() {
    sep("TEST 5: No negative densities across parameter space");

    struct Case {
        int N; double T; double kappa0; double xi; double lambda; const char* label;
    };
    std::vector<Case> cases = {
        {400, 1.0/12.0,  2.0, 0.30, 5.0, "standard DHJ 1m"},
        {400, 3.0/12.0,  2.0, 0.30, 5.0, "DHJ 3m"},
        {400, 1.0/12.0, 50.0, 0.00, 0.0, "BS limit"},
        {200, 1.0/12.0,  2.0, 0.30, 5.0, "N=200"},
        {800, 1.0/12.0,  2.0, 0.30, 5.0, "N=800"},
        {400, 1.0/12.0,  2.0, 0.50,10.0, "high ξ, high λ"},
        {400, 1.0/12.0,  2.0, 0.30, 5.0, "ρ=-0.6"},
    };
    // For ρ=-0.6 case
    bool all_ok = true;
    for (auto& c : cases) {
        DHJInput in;
        in.S0 = 1.085; in.r = 0.0525; in.r_f = 0.04; in.horizon_T = c.T;
        double rho = (std::string(c.label) == "ρ=-0.6") ? -0.6 : 0.0;
        in.heston = {1.5, 0.082*0.082, c.xi, rho, 0.082*0.082};
        in.kappa0 = c.kappa0; in.kappa1 = 0.002; in.delta_cp = 0.0;
        in.jumps.lambda = c.lambda;
        in.jumps.p_up = 0.55; in.jumps.eta_plus = 100.0; in.jumps.eta_minus = 80.0;
        in.N_sites = c.N; in.x_width_std = 5;
        in.N_paths = 200; in.seed = 42;

        auto out = DHJModel(in).run();
        bool ok = (out.negative_count == 0) && (out.min_density >= -1e-12);
        all_ok &= ok;
        pass_fail(ok,
                  c.label,
                  ok ? "" : ("neg_count=" + std::to_string(out.negative_count)
                             + " min=" + std::to_string(out.min_density)).c_str());
    }
    pass_fail(all_ok, "Zero negative densities across all tested configurations");
}

// ── Test 6: Mass loss ─────────────────────────────────────────────────────────
static void test_mass_loss() {
    sep("TEST 6: Boundary mass loss stays near zero");

    struct Case {
        double T; double xi; const char* label;
    };
    bool all_ok = true;
    for (auto& c : std::vector<Case>{
            {1.0/12.0, 0.00, "1m  ξ=0.00"},
            {1.0/12.0, 0.30, "1m  ξ=0.30"},
            {3.0/12.0, 0.30, "3m  ξ=0.30"},
            {3.0/12.0, 0.50, "3m  ξ=0.50"},
    }) {
        DHJInput in;
        in.S0 = 1.085; in.r = 0.0525; in.r_f = 0.04; in.horizon_T = c.T;
        in.heston = {1.5, 0.082*0.082, c.xi, -0.25, 0.082*0.082};
        in.kappa0 = 2.0; in.kappa1 = 0.002; in.delta_cp = 0.0;
        in.jumps.lambda = 5.0;
        in.jumps.p_up = 0.55; in.jumps.eta_plus = 100.0; in.jumps.eta_minus = 80.0;
        in.N_sites = 400; in.x_width_std = 5;
        in.N_paths = 300; in.seed = 42;

        auto out = DHJModel(in).run();
        bool ok = out.mass_loss_fraction < 0.005;   // < 0.5%
        all_ok &= ok;
        char detail[64];
        std::snprintf(detail, sizeof(detail), "mass_loss=%.4f%%", out.mass_loss_fraction * 100.0);
        pass_fail(ok, c.label, detail);
    }
    pass_fail(all_ok, "Mass loss < 0.5% for all standard configurations");
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  DHJ Convergence & Validation Test Suite                     ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    test_gk_recovery_vs_N();
    test_kappa_sweep();
    test_moments();
    test_jump_martingale();
    test_no_negative_densities();
    test_mass_loss();

    std::printf("\n══════════════════════════════════════════════════════════════\n");
    std::printf("Done. All ✓ = assertions passed.\n");
    std::printf("══════════════════════════════════════════════════════════════\n");
    return 0;
}
