#pragma once
/*
 * DHJModel — Dirac-Heston-Jump hybrid (Monte Carlo outer / PDE inner)
 * ──────────────────────────────────────────────────────────────────────────────
 * Five layers of dynamics:
 *
 *  1. Heston stochastic variance:
 *       dv = κ_H(θ−v) dt + ξ√v dW_v          (CIR, Milstein)
 *
 *  2. Dirac-Telegraph price — independent component only:
 *       D(τ) = (1−ρ²)·v/2,  c(τ) = √((1−ρ²)·v)
 *       Heston decomposes dW_S = ρ·dW_v + √(1−ρ²)·dW⊥; the PDE models only
 *       the independent part dW⊥. Using full v would double-count ρ²·v.
 *
 *  3. Regime-dependent Dirac mass:
 *       κ(v) = κ₀ + κ₁/v   (more ballistic in calm markets, diffusive in crises)
 *
 *  4. Leverage-correlated drift (conditional on each variance path):
 *       Δx_lev = ρ·√v·ΔW_v = ρ·√v·z(τ)·√dt
 *       Applied as a linear-interpolation grid shift after each PDE step.
 *       This is mathematically equivalent to the ρ√v·dW_v drift term but
 *       unconditionally stable — no CFL constraint on the leverage magnitude.
 *       The PDE drift ν = r − v/2 − λ·ζ  remains small and uses central diff.
 *
 *  5. Kou double-exponential jumps: applied as grid-cell shifts at each step.
 *       Jump compensator λ·ζ is subtracted from ν to preserve E[S_T]=S₀·e^{rT}.
 *
 * Black-Scholes limit: κ₀→∞, κ₁=0, ξ=0, λ=0, ρ=0 → exact BS ✓
 * Heston limit:        κ₀→∞, κ₁=0, λ=0         → stochastic-vol BS ✓
 * Dirac limit:         ξ=0,  λ=0,  ρ=0          → deterministic Dirac ✓
 *
 * Algorithm:
 *   for path = 1..N_paths:
 *     sample variance path v(τ) via Milstein; record z[i] = ΔW_v/√dt
 *     initialise spinor ψ±(x,0) = Gaussian(θ_ic)
 *     for each time step i:
 *       D_i=v/2, c_i=√v, κ_i=min(κ₀+κ₁/v, 50), ν_i=r−v/2−λζ
 *       adaptive sub-step if v_i > v_ref; evolve ψ± via RK4
 *       apply leverage grid-shift Δx=ρ√v·z·√dt (unconditionally stable)
 *       apply Kou jumps (also as grid shifts)
 *     accumulate ψ± weighted by final L1-norm; skip empty paths
 *   weighted average → final distribution; compute moments and option prices
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include "fx_dirac.hpp"
#include "heston_variance.hpp"
#include "jump_process.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <string>

// ── Input ─────────────────────────────────────────────────────────────────────
struct DHJInput {
    // Market
    Real S0;
    Real r;           // risk-free rate
    Real horizon_T;   // years

    // Heston variance (layer 1)
    HestonParams heston;   // {kappa_H, theta, xi, rho, v0}

    // Dirac mass (layers 2-3)
    Real kappa0;      // base Dirac mass              (e.g. 1.0)
    Real kappa1;      // vol-dependent correction κ₁/v (e.g. 0.005)
    Real delta_cp;    // initial spinor asymmetry      (0 = BS IC)

    // Kou jumps (layer 5)
    JumpParams jumps; // {lambda, p_up, eta_plus, eta_minus}

    // Lattice
    int  N_sites;
    int  x_width_std;

    // Monte Carlo
    int      N_paths;
    unsigned seed;    // 0 = random_device
};

// ── Output ────────────────────────────────────────────────────────────────────
struct DHJOutput {
    std::vector<Real> prices;
    std::vector<Real> prob_dhj;   // DHJ density (per unit price)
    std::vector<Real> prob_bs;    // BS reference density
    Real mean_dhj;
    Real mean_bs;
    Real var_log_dhj;
    Real var_log_bs;
    Real call_dhj;
    Real call_bs;
    Real chiral_charge;
    Real avg_variance;    // realised mean variance across paths
    int  n_steps;
    int  n_paths;
};

// ── Model ─────────────────────────────────────────────────────────────────────
class DHJModel {
public:
    explicit DHJModel(const DHJInput& in) : in_(in) { validate(); }

    DHJOutput run() const {
        const int  N   = in_.N_sites     > 0 ? in_.N_sites     : 400;
        const int  hw  = in_.x_width_std > 0 ? in_.x_width_std : 6; // wider for stoch-vol paths
        const Real T   = in_.horizon_T;
        const Real v0  = in_.heston.v0;
        const Real sig0 = std::sqrt(v0);

        // ── Lattice ───────────────────────────────────────────────────────────
        Real x_half = hw * sig0 * std::sqrt(T) + 0.5;
        Real x_min  = -x_half, x_max = x_half;
        Real dx     = (x_max - x_min) / (N - 1);

        // ── Heston variance decomposition ──────────────────────────────────────
        // The Heston price shock decomposes as:
        //   dW_S = ρ·dW_v  +  √(1−ρ²)·dW⊥    (independent components)
        // The correlated part ρ·dW_v is captured by the leverage grid shift.
        // The PDE evolves only the INDEPENDENT component: D_⊥ = (1−ρ²)·v/2.
        // Using full v would double-count the correlated variance ρ²·v.
        Real rho2   = in_.heston.rho * in_.heston.rho;
        Real v_ind  = (1.0 - rho2);  // scale factor for independent component

        // ── CFL — use realistic worst-case vol: θ + 2·σ_v (stationary Heston) ──
        //   Stationary variance std dev: σ_v = ξ·√(θ/(2κ_H))  (Gamma distribution)
        Real sigma_v = in_.heston.xi
                       * std::sqrt(in_.heston.theta
                                   / (2.0 * std::max(in_.heston.kappa_H, 0.1)));
        Real v_ref  = in_.heston.theta + 2.0 * sigma_v;
        v_ref = std::max({v_ref, v0, in_.heston.theta, 1e-6});
        Real D_ref  = v_ind * v_ref / 2.0;   // independent component only
        Real c_ref  = std::sqrt(v_ind * v_ref);
        Real dt     = 0.45 * dx * dx / (2.0 * D_ref + c_ref * dx + 1e-15);
        int  nsteps = (int)std::ceil(T / dt);
        dt           = T / nsteps;

        // ν = r − v/2 − jump_comp stays small; central diff is fine.
        // Keep nu_max as a sanity clamp (should never trigger).
        Real nu_max = 0.5 * dx / dt;

        // ── RNG ───────────────────────────────────────────────────────────────
        unsigned seed = in_.seed != 0 ? in_.seed : std::random_device{}();
        std::mt19937_64 rng(seed);

        // ── Processes ─────────────────────────────────────────────────────────
        HestonVarianceProcess heston(in_.heston);
        KouJumpProcess        jumps(in_.jumps);
        Real jump_comp = in_.jumps.lambda * jumps.jump_compensator();

        // ── Initial condition ──────────────────────────────────────────────────
        Real init_width = dx * std::max(2.0, 0.1 * sig0 * std::sqrt(T) / dx);
        Real theta_ic   = std::max(0.01, std::min(0.99, 0.5 + 0.5 * in_.delta_cp));

        // ── Monte Carlo loop ───────────────────────────────────────────────────
        int N_paths = in_.N_paths > 0 ? in_.N_paths : 300;
        std::vector<Real> acc_plus(N, 0.0), acc_minus(N, 0.0);
        Real acc_chiral = 0.0, acc_var = 0.0;

        // Weight-based accumulation: instead of normalising each path then averaging,
        // accumulate weighted by path final norm so leaking paths contribute proportionally
        // (avoids catastrophic amplification when norm ≈ 0 on extreme paths).
        Real acc_weight = 0.0;   // sum of path norms for weighted average

        std::vector<Real> v_path, z_path;

        for (int path = 0; path < N_paths; ++path) {
            // 1. Sample variance path
            heston.evolve(nsteps, dt, v_path, z_path, rng);

            // 2. Initialise spinor
            SpinorField field(N, x_min, x_max);
            field.init_gaussian(0.0, init_width, theta_ic);

            Real sum_v = 0.0;
            Real sqrdt = std::sqrt(dt);

            for (int i = 0; i < nsteps; ++i) {
                Real vi = std::max(v_path[i], 1e-10);
                Real zi = z_path[i];    // N(0,1) normalised increment

                // 3. Dirac params for this step — INDEPENDENT variance component only.
                // The correlated part ρ·√v·dW_v is applied below as a grid shift.
                // Using v_ind·v prevents double-counting the ρ²·v variance.
                Real D_i   = v_ind * vi / 2.0;
                Real c_i   = std::sqrt(v_ind * vi);
                // κ₁/v is singular at v→0; cap at 50 to keep κ·h < 0.01 (RK4 stable).
                // When Feller condition is violated (2κθ < ξ²), v touches 0 frequently,
                // and κ₁/v would otherwise blow past RK4's |λh| < 2.785 stability limit.
                Real kd_i  = std::min(in_.kappa0 + (vi > 1e-8 ? in_.kappa1 / vi : 0.0),
                                      50.0);

                // ν = r − v/2 − λζ  (full v for drift; leverage applied as shift below)
                // Risk-neutral log-return drift is r − v/2 regardless of ρ decomposition.
                Real nu_i  = (in_.r - vi / 2.0) - jump_comp;
                nu_i = std::max(-nu_max, std::min(nu_max, nu_i));

                // Adaptive sub-stepping: if v_i > v_ref (independent component), CFL violated.
                Real dt_cfl_i = 0.45 * dx * dx / (2.0*D_i + c_i*dx + 1e-15);
                int  n_sub    = (dt_cfl_i >= dt) ? 1
                              : std::min(32, (int)std::ceil(dt / dt_cfl_i));
                Real h = dt / n_sub;

                DiracParams dp;
                dp.D = D_i; dp.c = c_i;
                dp.kappa = kd_i; dp.gamma = kd_i;
                dp.nu = nu_i; dp.r = in_.r; dp.dt = h;
                dp.upwind_nu = false;   // ν is small — central diff is fine

                WilsonDiracEvolver evolver(dp);
                for (int s = 0; s < n_sub; ++s)
                    evolver.step_rk4(field, h);

                // 4. Leverage shift: Δx = ρ·√v·ΔW_v = ρ·√v·z·√dt
                //    Applied as sub-cell linear-interpolation grid shift.
                //    Unconditionally stable — no CFL constraint.
                Real delta_x_lev = in_.heston.rho * std::sqrt(vi) * zi * sqrdt;
                KouJumpProcess::shift_field_interp(field, delta_x_lev);

                // 5. Kou jumps (once per nominal dt, not per sub-step)
                jumps.apply(field, dt, rng);

                sum_v += vi;
            }

            // Weighted accumulation: weight = path norm (probability retained on grid).
            // Paths where distribution has leaked off the domain contribute less weight,
            // preventing catastrophic amplification of nearly-empty fields.
            Real norm_v = field.total_norm();
            if (norm_v < 1e-15) continue;   // completely empty — skip

            acc_weight += norm_v;
            for (int j = 0; j < N; ++j) {
                acc_plus[j]  += field.plus(j);
                acc_minus[j] += field.minus(j);
            }
            acc_chiral += field.chiral_charge();
            acc_var    += sum_v / nsteps;
        }

        // ── Normalise by accumulated weight (weighted MC average) ─────────────
        Real inv_w = (acc_weight > 1e-15) ? 1.0 / acc_weight : 0.0;
        Real total = 0.0;
        for (int j = 0; j < N; ++j) {
            acc_plus[j]  *= inv_w;
            acc_minus[j] *= inv_w;
            total += (acc_plus[j] + acc_minus[j]) * dx;
        }
        Real inv_tot = (total > 1e-15) ? 1.0 / total : 0.0;
        // acc_var and acc_chiral: weight by path count (all paths contributed)
        Real inv_p = 1.0 / N_paths;

        // ── Build output ───────────────────────────────────────────────────────
        DHJOutput out;
        out.n_steps       = nsteps;
        out.n_paths       = N_paths;
        out.avg_variance  = acc_var    * inv_p;
        out.chiral_charge = acc_chiral * inv_p;

        out.prices.resize(N);
        out.prob_dhj.resize(N);
        out.prob_bs.resize(N);

        Real mean_d = 0.0, mean_b = 0.0;
        Real var_d  = 0.0, mu_x  = 0.0;
        Real call_d = 0.0;
        Real sig_bs = sig0;
        Real K      = in_.S0;

        for (int j = 0; j < N; ++j) {
            Real xj = x_min + j * dx;
            Real Sj = in_.S0 * std::exp(xj);

            Real p_x    = (acc_plus[j] + acc_minus[j]) * inv_tot;
            Real p_S    = p_x / Sj;
            Real p_logbs = BS::lognormal_pdf(Sj, in_.S0, sig_bs, in_.r, T) * Sj;

            out.prices[j]   = Sj;
            out.prob_dhj[j] = p_S;
            out.prob_bs[j]  = p_logbs / Sj;

            mean_d += Sj * p_x     * dx;
            mean_b += Sj * p_logbs * dx;
            mu_x   += xj * p_x     * dx;
            var_d  += xj * xj * p_x * dx;
            call_d += std::max(Sj - K, 0.0) * p_x * dx;
        }

        out.mean_dhj    = mean_d;
        out.mean_bs     = mean_b;
        out.var_log_dhj = var_d - mu_x * mu_x;
        out.var_log_bs  = sig_bs * sig_bs * T;
        out.call_dhj    = std::exp(-in_.r * T) * call_d;
        out.call_bs     = BS::call_price(in_.S0, K, sig_bs, in_.r, T);

        return out;
    }

private:
    DHJInput in_;
    void validate() const {
        if (in_.S0 <= 0)         throw std::invalid_argument("S0 must be > 0");
        if (in_.horizon_T <= 0)  throw std::invalid_argument("horizon_T must be > 0");
        if (in_.heston.v0 <= 0)  throw std::invalid_argument("v0 must be > 0");
        if (in_.heston.theta<=0) throw std::invalid_argument("theta must be > 0");
        if (std::abs(in_.heston.rho) >= 1.0)
            throw std::invalid_argument("|rho| must be < 1");
        if (in_.heston.xi < 0)   throw std::invalid_argument("xi must be >= 0");
    }
};
