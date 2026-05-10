#pragma once
#include <complex>
#include <array>
#include <cmath>
#include <cassert>

// ── Fundamental numeric types ─────────────────────────────────────────────────
using Real    = double;
using Complex = std::complex<Real>;

static constexpr Complex I_UNIT{0.0, 1.0};

// ── 2-component spinor ψ = (ψ₊, ψ₋)ᵀ ────────────────────────────────────────
// ψ₊: "bullish" / right-moving component
// ψ₋: "bearish" / left-moving component
struct Spinor2 {
    Complex u, d;  // upper / lower component

    Spinor2() : u(0), d(0) {}
    Spinor2(Complex u_, Complex d_) : u(u_), d(d_) {}

    Spinor2 operator+(const Spinor2& o) const { return {u+o.u, d+o.d}; }
    Spinor2 operator-(const Spinor2& o) const { return {u-o.u, d-o.d}; }
    Spinor2 operator*(Complex s)        const { return {u*s,   d*s};   }
    Spinor2 operator*(Real   s)         const { return {u*s,   d*s};   }
    Spinor2& operator+=(const Spinor2& o) { u+=o.u; d+=o.d; return *this; }

    // Spinor norm² = |ψ₊|² + |ψ₋|² (probability density at this site)
    Real norm2() const { return std::norm(u) + std::norm(d); }

    // Adjoint ψ†
    Spinor2 adjoint() const { return {std::conj(u), std::conj(d)}; }
};

inline Spinor2 operator*(Complex s, const Spinor2& p) { return p * s; }
inline Spinor2 operator*(Real   s, const Spinor2& p) { return p * s; }

// ── 2×2 complex matrix ────────────────────────────────────────────────────────
struct Mat2 {
    // row-major: m[row][col]
    Complex m[2][2];

    Mat2() { m[0][0]=m[0][1]=m[1][0]=m[1][1] = 0; }
    Mat2(Complex a, Complex b, Complex c, Complex d) {
        m[0][0]=a; m[0][1]=b; m[1][0]=c; m[1][1]=d;
    }

    static Mat2 identity() { return {1,0,0,1}; }
    static Mat2 zero()     { return {0,0,0,0}; }

    Mat2 operator+(const Mat2& o) const {
        return {m[0][0]+o.m[0][0], m[0][1]+o.m[0][1],
                m[1][0]+o.m[1][0], m[1][1]+o.m[1][1]};
    }
    Mat2 operator-(const Mat2& o) const {
        return {m[0][0]-o.m[0][0], m[0][1]-o.m[0][1],
                m[1][0]-o.m[1][0], m[1][1]-o.m[1][1]};
    }
    Mat2 operator*(Real s) const {
        return {m[0][0]*s, m[0][1]*s, m[1][0]*s, m[1][1]*s};
    }
    Mat2 operator*(Complex s) const {
        return {m[0][0]*s, m[0][1]*s, m[1][0]*s, m[1][1]*s};
    }
    Mat2 operator*(const Mat2& o) const {
        return {
            m[0][0]*o.m[0][0] + m[0][1]*o.m[1][0],
            m[0][0]*o.m[0][1] + m[0][1]*o.m[1][1],
            m[1][0]*o.m[0][0] + m[1][1]*o.m[1][0],
            m[1][0]*o.m[0][1] + m[1][1]*o.m[1][1]
        };
    }

    // Matrix-spinor product
    Spinor2 apply(const Spinor2& p) const {
        return { m[0][0]*p.u + m[0][1]*p.d,
                 m[1][0]*p.u + m[1][1]*p.d };
    }

    Mat2 dagger() const {   // conjugate transpose
        return {std::conj(m[0][0]), std::conj(m[1][0]),
                std::conj(m[0][1]), std::conj(m[1][1])};
    }
};
inline Mat2 operator*(Real s, const Mat2& M) { return M * s; }
inline Mat2 operator*(Complex s, const Mat2& M) { return M * s; }
