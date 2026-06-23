#pragma once
/*
 * HestonVariance — CIR variance process evolved via Milstein scheme
 * ──────────────────────────────────────────────────────────────────────────────
 *   dv = κ_H(θ − v) dt + ξ √v dW_v          (Heston CIR)
 *
 * Milstein correction (strong order 1.0 vs Euler's 0.5):
 *   v_{n+1} = v_n + κ_H(θ−v_n)dt + ξ√v_n ΔW + ξ²/4·(ΔW²−dt)
 *
 * Full truncation at v=0 (ensures v ≥ 0 without reflecting boundary).
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include "spinor_field.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

struct HestonParams {
    Real kappa_H;   // mean-reversion speed       (e.g. 1.5)
    Real theta;     // long-run variance           (e.g. 0.0067 ≈ 8.2%²)
    Real xi;        // vol-of-vol                  (e.g. 0.3)
    Real rho;       // leverage correlation ∈(-1,1) (e.g. −0.3 for FX)
    Real v0;        // initial variance            (e.g. sigma²)
};

class HestonVarianceProcess {
public:
    explicit HestonVarianceProcess(const HestonParams& p) : p_(p) {}

    // Evolve one variance trajectory of exactly N steps of size dt.
    // Stores v[0..N] and the normalised increments z[0..N-1] = ΔW_v/√dt.
    // Caller provides pre-allocated v (size N+1) and z (size N).
    void evolve(int N, Real dt,
                std::vector<Real>& v, std::vector<Real>& z,
                std::mt19937_64& rng) const
    {
        v.resize(N + 1);
        z.resize(N);
        std::normal_distribution<Real> norm(0.0, 1.0);
        Real sqrdt = std::sqrt(dt);

        v[0] = std::max(p_.v0, 0.0);
        for (int i = 0; i < N; ++i) {
            Real zi   = norm(rng);
            Real vi   = std::max(v[i], 0.0);          // full truncation before sqrt
            Real sqvi = std::sqrt(vi);
            Real dW   = zi * sqrdt;

            v[i+1] = vi
                   + p_.kappa_H * (p_.theta - vi) * dt
                   + p_.xi * sqvi * dW
                   + 0.25 * p_.xi * p_.xi * (dW*dW - dt);
            v[i+1] = std::max(v[i+1], 0.0);
            z[i]   = zi;
        }
    }

    const HestonParams& params() const { return p_; }

private:
    HestonParams p_;
};
