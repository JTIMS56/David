#pragma once
/*
 * FinancialDiracModel — FX prediction via telegraph/Dirac brane-world dynamics
 * ──────────────────────────────────────────────────────────────────────────────
 * Maps standard financial parameters to the Dirac brane-world:
 *
 *  Financial         Brane-world interpretation
 *  ────────────────  ──────────────────────────────────────────────────────
 *  σ (volatility)  → c  (speed of light; maximum price velocity on brane)
 *                    D = σ²/2  (Wilson diffusion = BS diffusion coefficient)
 *  κ (flip rate)   → Dirac mass; large κ ↔ overdamped ↔ Black-Scholes
 *                    κ=0 ↔ massless spinor ↔ pure ballistic price motion
 *  μ (drift)       → ν = μ−r−σ²/2  (log-drift gauge field)
 *  r (risk-free)   → scalar potential (discounting)
 *
 * Black-Scholes limit:
 *   κ → ∞  (with D = σ²/2 fixed) ← OVERDAMPED telegraph process
 *   ψ₊ ≈ ψ₋ → P/2 equilibrates rapidly
 *   ∂_τ P = D ∂_x² P − ν ∂_x P − r P  [exact BS Fokker-Planck]
 *
 * Finite-κ corrections (non-Gaussian Dirac regime):
 *   Extra diffusion:  δD  ≈ c²/(2κ) = σ²/(2κ)
 *   Effective total:  D_eff ≈ σ²/2 · (1 + 1/κ)
 *   Skewness:         S ∝ ν/(κ σ√T)
 *   Fat tails for:    κ < 1/T  (light spinor, long horizon)
 *
 * The model is numerically stable, probability-positive, and reduces
 * EXACTLY to Black-Scholes as κ → ∞.
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include "wilson_dirac.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

// ── Input ─────────────────────────────────────────────────────────────────────
struct FXDiracInput {
    Real S0;          // spot price
    Real sigma;       // annualised volatility
    Real mu;          // annualised real-world drift
    Real r;           // risk-free rate
    Real kappa;       // Dirac mass / flip rate  (kappa=0 → wave; kappa→∞ → BS)
    Real delta_cp;    // CP asymmetry (initial ψ₊ bias; 0 = symmetric = BS IC)
    Real horizon_T;   // horizon in years
    int  N_sites;     // lattice sites (0 → auto)
    int  x_width_std; // half-width in σ√T units (0 → 5)
};

// ── Output ────────────────────────────────────────────────────────────────────
struct FXDiracOutput {
    std::vector<Real> prices;       // S values
    std::vector<Real> prob_dirac;   // Dirac P(S,T) per unit price
    std::vector<Real> prob_bs;      // BS   P(S,T) per unit price
    Real mean_dirac;
    Real mean_bs;
    Real var_log_dirac;   // Var[log(S_T/S_0)] from Dirac
    Real var_log_bs;      // σ²T  (analytical)
    Real call_dirac;      // discounted ATM call (K=S0)
    Real call_bs;         // BS ATM call
    Real chiral_charge;   // Q₅ = ∫(ψ₊−ψ₋)dx  (sentiment)
    int  n_steps;
};

// ── Black-Scholes reference ───────────────────────────────────────────────────
namespace BS {

inline Real norm_cdf(Real x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

inline Real lognormal_pdf(Real S, Real S0, Real sigma, Real r, Real T) {
    if (S <= 0 || T <= 0) return 0.0;
    Real log_S  = std::log(S / S0);
    Real mu_ln  = (r - 0.5 * sigma * sigma) * T;
    Real sig_ln = sigma * std::sqrt(T);
    Real z      = (log_S - mu_ln) / sig_ln;
    return std::exp(-0.5 * z * z) / (sig_ln * S * std::sqrt(2.0 * M_PI));
}

inline Real call_price(Real S0, Real K, Real sigma, Real r, Real T) {
    if (T <= 0) return std::max(S0 - K, 0.0);
    Real sig_sq_T = sigma * std::sqrt(T);
    Real d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / sig_sq_T;
    Real d2 = d1 - sig_sq_T;
    return S0 * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

} // namespace BS

// ── Model ─────────────────────────────────────────────────────────────────────
class FinancialDiracModel {
public:
    explicit FinancialDiracModel(const FXDiracInput& in) : in_(in) { validate(); }

    FXDiracOutput run() const {
        // ── Lattice setup ─────────────────────────────────────────────────────
        int  N   = in_.N_sites > 0 ? in_.N_sites : 400;
        int  hw  = in_.x_width_std > 0 ? in_.x_width_std : 5;
        Real T   = in_.horizon_T;
        Real sig = in_.sigma;

        Real x_half = hw * sig * std::sqrt(T)
                    + std::abs(in_.mu - in_.r) * T + 0.5;
        Real x_min = -x_half, x_max = x_half;
        Real dx    = (x_max - x_min) / (N - 1);

        // ── Dirac parameters ─────────────────────────────────────────────────
        //  D = σ²/2  (Wilson term provides the BS diffusion)
        //  c = σ     (wave speed = volatility)
        //  κ = gamma = user-supplied flip rate
        //  ν = risk-neutral log-drift
        Real D   = 0.5 * sig * sig;
        Real c   = sig;
        // Risk-neutral log-price drift: for Q-measure use mu=r → ν = r - σ²/2.
        // Setting mu=r in the input gives the standard BS risk-neutral dynamics;
        // any other mu gives a real-world measure prediction.
        Real nu  = in_.mu - 0.5 * sig * sig;

        DiracParams p;
        p.D     = D;
        p.c     = c;
        p.kappa = in_.kappa;
        p.gamma = in_.kappa;   // γ = κ ensures total-P conservation (undiscounted)
        p.nu    = nu;
        p.r     = in_.r;
        p.dt    = 0.45 * dx * dx / (2.0 * D + c * dx + 1e-15);  // CFL

        // ── Initial state ─────────────────────────────────────────────────────
        // Symmetric (θ=0.5): exact BS initial condition
        // Asymmetric (θ≠0.5): encodes initial momentum bias (CP violation)
        SpinorField field(N, x_min, x_max);
        Real init_width = dx * std::max(2.0, 0.1 * sig * std::sqrt(T) / dx);
        Real theta = 0.5 + 0.5 * in_.delta_cp;   // ∈ [0,1]
        theta = std::max(0.01, std::min(0.99, theta));
        field.init_gaussian(0.0, init_width, theta);

        // ── Evolve ────────────────────────────────────────────────────────────
        WilsonDiracEvolver evolver(p);
        int n_steps = evolver.evolve(field, T);

        // ── Normalise against numerical drift ────────────────────────────────
        // (discounting means norm should decay as e^{-rT}, re-scale for density)
        Real norm = field.total_norm();

        // ── Extract output ────────────────────────────────────────────────────
        FXDiracOutput out;
        out.n_steps = n_steps;
        out.prices.resize(N);
        out.prob_dirac.resize(N);
        out.prob_bs.resize(N);

        // All integrals are done in log-price space (variable x = ln S/S0):
        //   E[f(S)] = ∫ f(S(x)) · P_log(x) dx  ≈  Σ_j f(S_j) · p_x_j · dx
        // where P_log = P_S(S) · S  (change of variables, units: 1/log-price).
        //
        // The output prob_dirac[j] stores P_S = P_log / S  (per unit price, for
        // plotting against BS reference).
        //
        // KEY: do NOT mix P_log and P_S in the same integrand.  Using p_S * dS
        // with dS ≈ S_j * dx causes p_S * dS = p_x (no dx weight), giving
        // E[S]/dx rather than E[S].  Use p_x * dx throughout.
        Real mean_d = 0.0, mean_b = 0.0;
        Real var_d  = 0.0, mu_x  = 0.0;
        Real call_d = 0.0;
        Real K = in_.S0;

        for (int j = 0; j < N; ++j) {
            Real xj = field.x(j);
            Real Sj = in_.S0 * std::exp(xj);

            // P_log(x) — log-price density (sums to 1 over x with weight dx)
            Real p_x = (norm > 1e-15) ? field.density(j) / norm : 0.0;

            // P_S(S) — price density (per unit price); for plotting only
            Real p_S = p_x / Sj;   // P_S = P_log / S  (Jacobian dx/dS = 1/S)

            // BS log-price density: P_log_bs = P_S_bs * S
            Real p_log_bs = BS::lognormal_pdf(Sj, in_.S0, sig, in_.r, T) * Sj;

            out.prices[j]     = Sj;
            out.prob_dirac[j] = p_S;                        // P_S (per unit price)
            out.prob_bs[j]    = p_log_bs / Sj;              // = BS::lognormal_pdf

            // All integrals use dx weight (correct quadrature in log-price space)
            mean_d += Sj * p_x     * dx;                    // E_D[S]
            mean_b += Sj * p_log_bs * dx;                   // E_BS[S] = ∫ S·P_log dx
            mu_x   += xj * p_x     * dx;                    // E_D[x]
            var_d  += xj * xj * p_x * dx;                   // E_D[x²]
            call_d += std::max(Sj - K, 0.0) * p_x     * dx; // E_D[max(S-K,0)]
        }

        out.mean_dirac    = mean_d;
        out.mean_bs       = mean_b;
        out.var_log_dirac = var_d - mu_x * mu_x;
        out.var_log_bs    = sig * sig * T;
        out.call_dirac    = std::exp(-in_.r * T) * call_d;
        out.call_bs       = BS::call_price(in_.S0, K, sig, in_.r, T);
        out.chiral_charge = field.chiral_charge() / (norm + 1e-15);

        return out;
    }

private:
    FXDiracInput in_;
    void validate() const {
        if (in_.sigma <= 0)     throw std::invalid_argument("sigma must be > 0");
        if (in_.horizon_T <= 0) throw std::invalid_argument("horizon_T must be > 0");
        if (in_.S0 <= 0)        throw std::invalid_argument("S0 must be > 0");
    }
};
