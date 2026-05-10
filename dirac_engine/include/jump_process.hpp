#pragma once
/*
 * KouJumps — Double-exponential (Kou 2002) jump-diffusion overlay
 * ──────────────────────────────────────────────────────────────────────────────
 * Poisson arrivals at rate λ (per year); each jump Y in log-price space:
 *
 *   f_Y(y) = p·η₊·e^{−η₊ y}·1{y>0}  +  (1−p)·η₋·e^{η₋ y}·1{y<0}
 *
 * where p is the probability of an upward jump.
 *
 * Typical FX calibration:
 *   λ = 3–5 (jumps/year)   p  = 0.55 (slight upward bias)
 *   η₊ = 20 (small up)     η₋ = 15  (slightly larger down — crash asymmetry)
 *
 * Risk-neutral correction (martingale adjustment):
 *   The mean compensator ζ = E[e^Y−1] = p·η₊/(η₊−1) + (1−p)·η₋/(η₋+1) − 1
 *   must be subtracted from ν to keep E[S_T]=S₀·exp(rT).
 *   Call jump_compensator() to get ζ and subtract λ·ζ from ν.
 *
 * Implementation:
 *   At each PDE time step of size dt, Poisson(λ·dt) jumps are sampled.
 *   Each jump shifts the spinor field on the lattice by rounding Y to the
 *   nearest grid cell (integer shift). Sub-cell precision is negligible when
 *   dt is small (E[|Y|] ≈ 1/η ~ 0.05 log-units ≈ 16 grid cells for dx≈0.003).
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include "spinor_field.hpp"
#include <random>
#include <cmath>
#include <algorithm>

struct JumpParams {
    Real lambda;     // arrival rate (jumps/year); 0 = no jumps
    Real p_up;       // probability of upward jump  ∈ (0, 1)
    Real eta_plus;   // upward jump rate  (mean up-jump = 1/η₊)
    Real eta_minus;  // downward jump rate (mean down-jump = 1/η₋)
};

class KouJumpProcess {
public:
    explicit KouJumpProcess(const JumpParams& p) : p_(p) {}

    // E[e^Y − 1] — risk-neutral compensator; subtract λ·this from ν
    Real jump_compensator() const {
        if (p_.lambda <= 0.0 || p_.eta_plus <= 1.0) return 0.0;
        Real up   = p_.p_up       * p_.eta_plus  / (p_.eta_plus  - 1.0);
        Real down = (1.0-p_.p_up) * p_.eta_minus / (p_.eta_minus + 1.0);
        return up + down - 1.0;
    }

    // Apply Poisson jumps to spinor field over one time step dt.
    // Returns number of jumps applied.
    int apply(SpinorField& field, Real dt, std::mt19937_64& rng) const {
        if (p_.lambda <= 0.0) return 0;

        std::poisson_distribution<int>     poisson(p_.lambda * dt);
        std::uniform_real_distribution<Real> unif(0.0, 1.0);
        std::exponential_distribution<Real>  exp_up(p_.eta_plus);
        std::exponential_distribution<Real>  exp_dn(p_.eta_minus);

        int n = poisson(rng);
        for (int k = 0; k < n; ++k) {
            Real Y = (unif(rng) < p_.p_up) ? exp_up(rng) : -exp_dn(rng);
            shift_field(field, Y);
        }
        return n;
    }

    // Continuous sub-cell shift via linear interpolation — safe for any Y.
    // Used for leverage correlation (ρ·√v·ΔW_v) and for jumps when high accuracy needed.
    static void shift_field_interp(SpinorField& field, Real Y) {
        if (std::abs(Y) < 1e-14) return;
        int  N  = field.size();
        Real dx = field.dx();

        // Decompose into integer + fractional parts
        Real frac_raw = Y / dx;
        int  ns   = (int)std::floor(frac_raw);   // floor → frac ∈ [0,1)
        Real frac = frac_raw - ns;                // ∈ [0, 1)

        auto& rp = field.raw_plus();
        auto& rm = field.raw_minus();

        // Scratch buffers
        std::vector<Real> tp(N, 0.0), tm(N, 0.0);

        // Shift right by ns + frac: new[j] = (1-frac)*old[j-ns] + frac*old[j-ns-1]
        for (int j = 0; j < N; ++j) {
            int src  = j - ns;       // integer-shifted source
            int src1 = src - 1;      // sub-cell source
            Real v0 = (src  >= 0 && src  < N) ? rp[src]  : 0.0;
            Real v1 = (src1 >= 0 && src1 < N) ? rp[src1] : 0.0;
            tp[j] = (1.0 - frac) * v0 + frac * v1;
            Real m0 = (src  >= 0 && src  < N) ? rm[src]  : 0.0;
            Real m1 = (src1 >= 0 && src1 < N) ? rm[src1] : 0.0;
            tm[j] = (1.0 - frac) * m0 + frac * m1;
        }
        rp = tp;  rm = tm;
    }

    const JumpParams& params() const { return p_; }

private:
    // Translate spinor field by Y log-price units (nearest-grid-cell shift).
    void shift_field(SpinorField& field, Real Y) const {
        shift_field_interp(field, Y);   // use interp for all shifts
    }

    JumpParams p_;
};
