#pragma once
/*
 * Wilson-Dirac Evolution — Telegraph / Goldstein-Kac process on a price lattice
 * ──────────────────────────────────────────────────────────────────────────────
 * Evolves (ψ₊, ψ₋) forward in Euclidean time τ (= real financial time):
 *
 *   ∂_τ ψ₊(j) = D·Δ²ψ₊(j) − c·∇⁻ψ₊(j) − ν·∇⁰ψ₊(j) − κ·ψ₊(j) + γ·ψ₋(j) − r·ψ₊(j)
 *   ∂_τ ψ₋(j) = D·Δ²ψ₋(j) + c·∇⁺ψ₋(j) − ν·∇⁰ψ₋(j) − κ·ψ₋(j) + γ·ψ₊(j) − r·ψ₋(j)
 *
 * where:
 *   Δ²f(j)   = [f(j+1) − 2f(j) + f(j−1)] / dx²   (second-order diffusion)
 *   ∇⁻f(j)   = [f(j) − f(j−1)] / dx               (backward / upwind for ψ₊)
 *   ∇⁺f(j)   = [f(j+1) − f(j)] / dx               (forward / upwind for ψ₋)
 *   ∇⁰f(j)   = [f(j+1) − f(j−1)] / (2 dx)         (central for drift)
 *
 * Sign convention: drift ν enters as −ν·∇⁰ (continuity equation sign).
 *   The Fokker-Planck advection of density f with velocity ν > 0 is:
 *   ∂_t f = −∂_x(νf) = −ν ∂_x f  →  moves peak rightward.
 *   Each spinor component picks up −ν ∂_x, so P = ψ₊+ψ₋ also gets −ν ∂_x P.
 *
 * Parameters:
 *   D  = σ²/2        diffusion coefficient  (Wilson term = BS diffusion)
 *   c  = σ           wave velocity          (Dirac "speed of light" on brane)
 *   κ  = flip rate   momentum damping        (large κ → overdamped → BS)
 *   γ  = κ           chiral mixing rate      (set γ=κ to conserve total P)
 *   ν  = μ−σ²/2     log-price drift (use μ=r for risk-neutral Q-measure)
 *   r  = risk-free rate (discounting)
 *
 * Black-Scholes limit  (κ = γ → ∞  with  D = σ²/2  fixed):
 *   ψ₊ ≈ ψ₋ ≈ P/2,  so the current term c·∂_x(ψ₋−ψ₊) → 0
 *   ∂_τ P = D·∂²_x P − ν·∂_x P − r·P   ← EXACT Fokker-Planck / BS ✓
 *
 * Finite-κ corrections:
 *   δD   ≈ c²/(2κ)   (extra diffusion from finite-speed propagation)
 *   Skew ∝ 1/κ       (third-cumulant from asymmetric momentum)
 *   These create the non-Gaussian fat tails absent in BS.
 *
 * Stability (CFL for explicit Euler):
 *   dt ≤ dx² / (2D + c·dx)   — automatically enforced in evolve()
 *
 * Integrator: 4th-order Runge-Kutta
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include "spinor_field.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

struct DiracParams {
    Real D;           // diffusion coeff  = σ²/2
    Real c;           // wave speed       = σ
    Real kappa;       // flip / damping rate
    Real gamma;       // chiral mixing rate (= kappa for probability conservation)
    Real nu;          // log-return drift  = μ − σ²/2
    Real r;           // risk-free rate (discounting)
    Real dt;          // time step (used as max step; CFL enforced internally)
    bool upwind_nu = false; // use upwind (not central diff) for ν — required when |ν|*dt/dx > 0.5
};

class WilsonDiracEvolver {
public:
    explicit WilsonDiracEvolver(const DiracParams& p) : p_(p) {}

    // Compute RHS: stores dψ₊/dτ in dp and dψ₋/dτ in dm (same size as field)
    void rhs(const SpinorField& f,
             std::vector<Real>& dp, std::vector<Real>& dm) const
    {
        int  N  = f.size();
        Real dx = f.dx();
        Real inv_dx  = 1.0 / dx;
        Real inv_dx2 = 1.0 / (dx * dx);

        dp.resize(N);
        dm.resize(N);

        for (int j = 0; j < N; ++j) {
            // Neighbours (Dirichlet BC: zero outside domain)
            Real pp = (j+1 < N) ? f.plus(j+1)  : 0.0;
            Real pc =              f.plus(j);
            Real pm = (j-1 >= 0) ? f.plus(j-1)  : 0.0;

            Real mp = (j+1 < N) ? f.minus(j+1) : 0.0;
            Real mc =              f.minus(j);
            Real mm = (j-1 >= 0) ? f.minus(j-1) : 0.0;

            // ── ψ₊ (right-mover: advects in +x direction)
            Real diff_p    = p_.D * (pp - 2.0*pc + pm) * inv_dx2;
            // Wave term: central differences (O(dx²)) — avoids the O(dx) numerical
            // diffusion of first-order upwind while remaining stable under CFL.
            // The Wilson diffusion term already damps unphysical high-k modes.
            Real wave_p    = -p_.c * (pp - pm) * (0.5 * inv_dx);   // −c ∇⁰ψ₊
            // Drift −ν ∂_x ψ: central diff (2nd order) or upwind (1st, stable for large ν)
            Real drift_p, drift_m;
            if (!p_.upwind_nu) {
                drift_p = -p_.nu * (pp - pm) * (0.5 * inv_dx);   // central ∇⁰
            } else if (p_.nu >= 0) {                               // rightward: backward ∇⁻
                drift_p = -p_.nu * (pc - pm) * inv_dx;
            } else {                                               // leftward: forward ∇⁺
                drift_p = -p_.nu * (pp - pc) * inv_dx;
            }
            Real mass_p    = -p_.kappa * pc + p_.gamma * mc;
            Real disc_p    = -p_.r   * pc;
            dp[j] = diff_p + wave_p + drift_p + mass_p + disc_p;

            // ── ψ₋ (left-mover: advects in −x direction)
            Real diff_m    = p_.D * (mp - 2.0*mc + mm) * inv_dx2;
            Real wave_m    = +p_.c * (mp - mm) * (0.5 * inv_dx);   // +c ∇⁰ψ₋
            if (!p_.upwind_nu) {
                drift_m = -p_.nu * (mp - mm) * (0.5 * inv_dx);   // central ∇⁰
            } else if (p_.nu >= 0) {
                drift_m = -p_.nu * (mc - mm) * inv_dx;
            } else {
                drift_m = -p_.nu * (mp - mc) * inv_dx;
            }
            Real mass_m    = -p_.kappa * mc + p_.gamma * pc;
            Real disc_m    = -p_.r   * mc;
            dm[j] = diff_m + wave_m + drift_m + mass_m + disc_m;
        }
    }

    // One RK4 step with step size h
    void step_rk4(SpinorField& field, Real h) const {
        int N = field.size();
        Real xmin = field.x_min(), xmax = field.x_min() + (N-1)*field.dx();
        std::vector<Real> k1p, k1m, k2p, k2m, k3p, k3m, k4p, k4m;

        // k1
        rhs(field, k1p, k1m);

        // k2
        SpinorField f2(N, xmin, xmax);
        for (int j=0;j<N;++j){
            f2.plus(j)  = field.plus(j)  + 0.5*h*k1p[j];
            f2.minus(j) = field.minus(j) + 0.5*h*k1m[j];
        }
        rhs(f2, k2p, k2m);

        // k3
        SpinorField f3(N, xmin, xmax);
        for (int j=0;j<N;++j){
            f3.plus(j)  = field.plus(j)  + 0.5*h*k2p[j];
            f3.minus(j) = field.minus(j) + 0.5*h*k2m[j];
        }
        rhs(f3, k3p, k3m);

        // k4
        SpinorField f4(N, xmin, xmax);
        for (int j=0;j<N;++j){
            f4.plus(j)  = field.plus(j)  + h*k3p[j];
            f4.minus(j) = field.minus(j) + h*k3m[j];
        }
        rhs(f4, k4p, k4m);

        // Combine
        Real h6 = h / 6.0;
        for (int j = 0; j < N; ++j) {
            field.plus(j)  += h6*(k1p[j] + 2.0*k2p[j] + 2.0*k3p[j] + k4p[j]);
            field.minus(j) += h6*(k1m[j] + 2.0*k2m[j] + 2.0*k3m[j] + k4m[j]);
        }
    }

    // Advance by total time T; return number of steps taken
    int evolve(SpinorField& field, Real T) const {
        Real dx = field.dx();
        // CFL: dt ≤ dx²/(2D + c·dx)
        Real dt_cfl = 0.45 * dx*dx / (2.0*p_.D + std::abs(p_.c)*dx + 1e-15);
        Real dt_use = std::min(p_.dt, dt_cfl);
        if (dt_use <= 0.0) dt_use = dt_cfl;

        int steps = 0;
        Real t = 0.0;
        while (t < T - 1e-14) {
            Real h = std::min(dt_use, T - t);
            step_rk4(field, h);
            t += h;
            ++steps;
        }
        return steps;
    }

    const DiracParams& params() const { return p_; }

private:
    DiracParams p_;
};
