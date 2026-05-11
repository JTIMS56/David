#pragma once
/*
 * SpinorField — two real probability-flux components on a log-price lattice
 * ──────────────────────────────────────────────────────────────────────────────
 * ψ₊(x,τ) = right-moving probability flux  (bullish momentum)
 * ψ₋(x,τ) = left-moving probability flux   (bearish momentum)
 * P(x,τ)  = ψ₊ + ψ₋                       (total probability density)
 *
 * Both ψ₊ and ψ₋ are REAL and NON-NEGATIVE throughout.  There are no
 * complex phases: this is a Euclidean (imaginary-time) formulation where
 * the spinor components are genuine probability densities, not quantum
 * amplitudes.
 *
 * Lattice: x_j = x_min + j · dx,  j = 0 … N-1
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>

using Real = double;

class SpinorField {
public:
    SpinorField(int N, Real x_min, Real x_max)
        : N_(N), x_min_(x_min), dx_((x_max - x_min) / (N - 1)),
          plus_(N, 0.0), minus_(N, 0.0)
    {
        if (N < 8) throw std::invalid_argument("Need ≥ 8 lattice sites");
    }

    int  size()  const { return N_; }
    Real dx()    const { return dx_; }
    Real x_min() const { return x_min_; }
    Real x(int j) const { return x_min_ + j * dx_; }

    Real& plus(int j)        { return plus_[j]; }
    Real  plus(int j)  const { return plus_[j]; }
    Real& minus(int j)       { return minus_[j]; }
    Real  minus(int j) const { return minus_[j]; }

    // Total probability density at site j
    Real density(int j) const { return plus_[j] + minus_[j]; }

    // Chiral density (spinor current proxy): Q₅(j) = ψ₊ - ψ₋
    Real chiral(int j) const { return plus_[j] - minus_[j]; }

    // ── Initialisers ──────────────────────────────────────────────────────────

    // Equal-weight Gaussian centred at x0.
    // Setting plus = minus = G/2 is the EXACT initial state for Black-Scholes
    // (symmetric sector: no initial momentum bias).
    void init_symmetric_gaussian(Real x0, Real width) {
        Real norm = 0.0;
        for (int j = 0; j < N_; ++j) {
            Real d = x(j) - x0;
            Real g = std::exp(-d*d / (2.0 * width * width));
            plus_[j] = minus_[j] = 0.5 * g;
            norm += g * dx_;
        }
        Real inv = 1.0 / norm;
        for (int j = 0; j < N_; ++j) { plus_[j] *= inv; minus_[j] *= inv; }
    }

    // Asymmetric Gaussian: theta∈[0,1] fraction goes into ψ₊
    // theta=0.5 → symmetric (BS); theta>0.5 → bullish bias
    void init_gaussian(Real x0, Real width, Real theta = 0.5) {
        Real norm = 0.0;
        for (int j = 0; j < N_; ++j) {
            Real d = x(j) - x0;
            Real g = std::exp(-d*d / (2.0 * width * width));
            plus_[j]  = theta       * g;
            minus_[j] = (1.0-theta) * g;
            norm += g * dx_;
        }
        Real inv = 1.0 / norm;
        for (int j = 0; j < N_; ++j) { plus_[j] *= inv; minus_[j] *= inv; }
    }

    // Discrete delta initial condition: all probability mass at the single
    // lattice site closest to x0.  Useful for convergence studies; in production
    // use init_gaussian with width=dx to avoid slow integral convergence near S₀.
    // theta∈[0,1]: fraction of mass into ψ₊ (0.5 = BS symmetric).
    void init_delta(Real x0, Real theta = 0.5) {
        std::fill(plus_.begin(),  plus_.end(),  0.0);
        std::fill(minus_.begin(), minus_.end(), 0.0);
        // Nearest grid site (clamped to interior)
        int j0 = static_cast<int>(std::round((x0 - x_min_) / dx_));
        j0 = std::max(1, std::min(N_ - 2, j0));
        // ψ₊[j0] + ψ₋[j0] = 1/dx  so that ∫ P dx = 1
        plus_[j0]  = theta       / dx_;
        minus_[j0] = (1.0-theta) / dx_;
    }

    // ── Observables ───────────────────────────────────────────────────────────

    // ∫ P(x) dx  (should stay ≈ 1 for undiscounted evolution)
    Real total_norm() const {
        Real s = 0.0;
        for (int j = 0; j < N_; ++j) s += density(j);
        return s * dx_;
    }

    // ∫ (ψ₊ - ψ₋) dx  (net momentum / market sentiment)
    Real chiral_charge() const {
        Real s = 0.0;
        for (int j = 0; j < N_; ++j) s += chiral(j);
        return s * dx_;
    }

    // ⟨x⟩ = ∫ x P(x) dx
    Real mean_log_price() const {
        Real s = 0.0;
        for (int j = 0; j < N_; ++j) s += x(j) * density(j);
        return s * dx_;
    }

    // Var[x] = ⟨x²⟩ - ⟨x⟩²
    Real log_variance() const {
        Real mu = mean_log_price();
        Real s = 0.0;
        for (int j = 0; j < N_; ++j) {
            Real d = x(j) - mu;
            s += d * d * density(j);
        }
        return s * dx_;
    }

    // Fill caller-allocated arrays (length N)
    void extract(Real* x_out, Real* prob_out) const {
        for (int j = 0; j < N_; ++j) {
            x_out[j]    = x(j);
            prob_out[j] = density(j);
        }
    }

    void copy_from(const SpinorField& o) {
        assert(N_ == o.N_);
        plus_  = o.plus_;
        minus_ = o.minus_;
    }

    std::vector<Real>& raw_plus()  { return plus_; }
    std::vector<Real>& raw_minus() { return minus_; }

private:
    int N_;
    Real x_min_, dx_;
    std::vector<Real> plus_, minus_;
};
