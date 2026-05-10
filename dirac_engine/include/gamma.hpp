#pragma once
/*
 * Gamma matrices for 1+1-dimensional Minkowski space
 * ──────────────────────────────────────────────────────────────────────────────
 * Clifford algebra:  { γ^μ, γ^ν } = 2 g^{μν}   with   g = diag(+1, -1)
 *
 * Dirac representation (standard):
 *   γ⁰ = σ_z = [[1, 0], [0,-1]]          (timelike, β matrix)
 *   γ¹ = i σ_y = [[0, 1], [-1, 0]]        (spacelike)
 *
 * Derived:
 *   α ≡ γ⁰γ¹ = [[0,1],[1,0]] = σ_x       (Dirac alpha, kinetic term)
 *   β ≡ γ⁰   = σ_z                        (Dirac beta,  mass term)
 *   γ⁵ = γ⁰γ¹ in 1+1D = σ_x              (chirality)
 *
 * Financial brane-world interpretation:
 *   α eigenspinors  ↔  pure momentum states (trending market)
 *   β eigenspinors  ↔  pure mass states     (mean-reverting market)
 *   γ⁵ = +1 sector ↔  right-chirality (bullish bias)
 *   γ⁵ = -1 sector ↔  left-chirality  (bearish bias)
 * ──────────────────────────────────────────────────────────────────────────────
 */
#include "types.hpp"

namespace Gamma {

// γ⁰ = σ_z  (timelike / β / mass matrix)
inline Mat2 g0() { return { 1, 0, 0, -1 }; }

// γ¹ = iσ_y  (spacelike)
inline Mat2 g1() { return { 0, 1, -1, 0 }; }

// α = γ⁰γ¹ = σ_x  (kinetic coupling between ψ₊ and ψ₋)
inline Mat2 alpha() { return { 0, 1, 1, 0 }; }

// β = γ⁰ = σ_z  (mass term; ψ₊ ↔ +m, ψ₋ ↔ -m)
inline Mat2 beta()  { return g0(); }

// γ⁵ = iγ⁰γ¹ in 1+1D = iσ_x (chirality)
inline Mat2 g5() { return alpha() * I_UNIT; }

// Projection operators P± = (1 ± γ⁵)/2  (chiral projectors)
inline Mat2 P_plus()  { return (Mat2::identity() * 0.5) + (g5() * (0.5)); }
inline Mat2 P_minus() { return (Mat2::identity() * 0.5) - (g5() * (0.5)); }

// Wilson projectors  P^W_±(μ) = (1 ± γ^μ) / 2
// Used in Wilson-Dirac to remove doublers
inline Mat2 W_time_plus()   { return {1, 0, 0, 0}; }  // (1+γ⁰)/2
inline Mat2 W_time_minus()  { return {0, 0, 0, 1}; }  // (1-γ⁰)/2
inline Mat2 W_space_plus()  { return {0.5,  0.5, 0.5,  0.5}; }  // (1+σ_x)/2
inline Mat2 W_space_minus() { return {0.5, -0.5,-0.5,  0.5}; }  // (1-σ_x)/2

// Verify anticommutation: {γ⁰, γ¹} = 0  (should be zero matrix)
inline bool verify_clifford() {
    Mat2 ac = g0()*g1() + g1()*g0();
    return (std::abs(ac.m[0][0]) + std::abs(ac.m[0][1]) +
            std::abs(ac.m[1][0]) + std::abs(ac.m[1][1])) < 1e-12;
}

} // namespace Gamma
