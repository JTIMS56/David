"""
DiracPredictor — Python ctypes wrapper for the C++ Dirac FX engine.

Exposes two models:
  predict()     — pure 1+1D Dirac/Telegraph model (deterministic vol)
  predict_dhj() — full Dirac-Heston-Jump hybrid (MC-PDE, stochastic vol + jumps)
"""
import ctypes
import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Path resolution ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_SO   = os.path.join(_HERE, "..", "dirac_engine", "build", "dirac_fx.so")

# ── Market-pair default parameters ────────────────────────────────────────────
_PAIR_PARAMS: dict[str, dict] = {
    "EURUSD": {"sigma": 0.082, "r": 0.0525},
    "GBPUSD": {"sigma": 0.098, "r": 0.0525},
    "USDJPY": {"sigma": 0.075, "r": 0.0525},
    "AUDUSD": {"sigma": 0.110, "r": 0.0435},
    "USDCAD": {"sigma": 0.078, "r": 0.0525},
    "EURGBP": {"sigma": 0.062, "r": 0.0400},
    "NZDUSD": {"sigma": 0.115, "r": 0.0550},
    "USDCHF": {"sigma": 0.070, "r": 0.0525},
}

_DEFAULT_KAPPA     = 2.0   # moderate Dirac mass (between wave and diffusion)
_DEFAULT_DELTA_CP  = 0.0   # symmetric initial condition = BS IC
_DEFAULT_N_SITES   = 400


@dataclass
class DHJPrediction:
    prices:       list[float]
    prob_dhj:     list[float]
    prob_bs:      list[float]
    mean_dhj:     float
    mean_bs:      float
    var_log_dhj:  float
    var_log_bs:   float
    call_dhj:     float
    call_bs:      float
    chiral_charge: float
    avg_variance: float
    n_steps:      int
    n_paths:      int
    horizon_days: float
    kappa0:       float
    xi:           float
    rho:          float
    jump_lambda:  float


@dataclass
class DiracPrediction:
    prices:      list[float]
    prob_dirac:  list[float]   # per-unit-price density (Dirac)
    prob_bs:     list[float]   # per-unit-price density (Black-Scholes)
    mean_dirac:  float
    mean_bs:     float
    var_log_dirac: float
    var_log_bs:    float
    call_dirac:  float
    call_bs:     float
    chiral_charge: float       # market sentiment proxy Q₅
    n_steps:     int
    kappa:       float
    horizon_days: float


class DiracPredictor:
    """Singleton-friendly wrapper; call DiracPredictor.instance() for shared use."""

    _instance: Optional["DiracPredictor"] = None

    def __init__(self) -> None:
        so_path = os.path.abspath(_SO)
        if not os.path.exists(so_path):
            raise FileNotFoundError(
                f"Dirac shared library not found at {so_path}. "
                "Run: cmake --build dirac_engine/build --parallel 4"
            )
        self._lib = ctypes.CDLL(so_path)
        self._setup_signatures()
        logger.info("DiracPredictor loaded %s", so_path)

    @classmethod
    def instance(cls) -> "DiracPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── C function signatures ──────────────────────────────────────────────────
    def _setup_signatures(self) -> None:
        dp = self._lib.dirac_predict
        dp.restype  = ctypes.c_int
        dp.argtypes = [
            ctypes.c_double,  # S0
            ctypes.c_double,  # sigma
            ctypes.c_double,  # mu
            ctypes.c_double,  # r
            ctypes.c_double,  # kappa
            ctypes.c_double,  # delta_cp
            ctypes.c_double,  # horizon_T
            ctypes.c_int,     # n_sites
            ctypes.POINTER(ctypes.c_double),  # prices_out
            ctypes.POINTER(ctypes.c_double),  # prob_dirac_out
            ctypes.POINTER(ctypes.c_double),  # prob_bs_out
            ctypes.c_int,                     # n_out
            ctypes.POINTER(ctypes.c_double),  # scalars_out [8]
        ]

        bt = self._lib.dirac_bs_convergence_test
        bt.restype  = ctypes.c_int
        bt.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]

        ver = self._lib.dirac_version
        ver.restype  = None
        ver.argtypes = [ctypes.c_char_p, ctypes.c_int]

        # ── DHJ C API ──────────────────────────────────────────────────────────
        dhj = self._lib.dhj_predict
        dhj.restype  = ctypes.c_int
        dhj.argtypes = [
            # Market
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            # Heston: kappa_H, theta, xi, rho, v0
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
            # Dirac: kappa0, kappa1, delta_cp
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            # Jumps: lambda, p_up, eta_plus, eta_minus
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            # Lattice + MC
            ctypes.c_int, ctypes.c_int, ctypes.c_uint,
            # Output
            ctypes.POINTER(ctypes.c_double),  # prices
            ctypes.POINTER(ctypes.c_double),  # prob_dhj
            ctypes.POINTER(ctypes.c_double),  # prob_bs
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),  # scalars [10]
        ]

        dver = self._lib.dhj_version
        dver.restype  = None
        dver.argtypes = [ctypes.c_char_p, ctypes.c_int]

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(
        self,
        pair: str,
        spot: float,
        horizon_days: float = 30.0,
        kappa: float        = _DEFAULT_KAPPA,
        delta_cp: float     = _DEFAULT_DELTA_CP,
        mu: Optional[float] = None,
        n_sites: int        = _DEFAULT_N_SITES,
    ) -> DiracPrediction:
        """
        Run the Dirac FX model for *pair* at spot price *spot*.

        Args:
            pair:          Currency pair key e.g. "EURUSD"
            spot:          Current spot price (S₀)
            horizon_days:  Forecast horizon in calendar days
            kappa:         Dirac mass / flip rate (2 = moderate; 0 = pure wave;
                           large → Black-Scholes)
            delta_cp:      Initial spinor asymmetry in [-1, 1]; 0 = BS IC
            mu:            Real-world drift override; None → use risk-free rate
                           (gives risk-neutral / Q-measure distribution)
            n_sites:       Lattice resolution (400 is default)

        Returns:
            DiracPrediction dataclass
        """
        params  = _PAIR_PARAMS.get(pair.upper(), {"sigma": 0.085, "r": 0.0500})
        sigma   = params["sigma"]
        r       = params["r"]
        mu_val  = r if mu is None else mu   # default: risk-neutral
        T       = horizon_days / 365.0

        n     = n_sites
        prices_arr     = (ctypes.c_double * n)()
        prob_dirac_arr = (ctypes.c_double * n)()
        prob_bs_arr    = (ctypes.c_double * n)()
        scalars_arr    = (ctypes.c_double * 8)()

        n_fill = self._lib.dirac_predict(
            spot, sigma, mu_val, r,
            kappa, delta_cp, T,
            n,
            prices_arr, prob_dirac_arr, prob_bs_arr,
            n,
            scalars_arr,
        )

        if n_fill < 0:
            raise RuntimeError(f"dirac_predict failed for pair={pair}")

        return DiracPrediction(
            prices      = list(prices_arr[:n_fill]),
            prob_dirac  = list(prob_dirac_arr[:n_fill]),
            prob_bs     = list(prob_bs_arr[:n_fill]),
            mean_dirac  = scalars_arr[0],
            mean_bs     = scalars_arr[1],
            var_log_dirac = scalars_arr[2],
            var_log_bs    = scalars_arr[3],
            call_dirac  = scalars_arr[4],
            call_bs     = scalars_arr[5],
            chiral_charge = scalars_arr[6],
            n_steps     = int(scalars_arr[7]),
            kappa       = kappa,
            horizon_days = horizon_days,
        )

    def bs_convergence_test(
        self,
        S0: float = 1.0,
        sigma: float = 0.082,
        r: float = 0.0525,
        T: float = 1.0 / 12.0,
        tol: float = 0.05,
    ) -> dict:
        """Verify κ→∞ limit gives BS call within *tol* relative error."""
        call_d = ctypes.c_double()
        call_b = ctypes.c_double()
        passed = self._lib.dirac_bs_convergence_test(
            S0, sigma, r, T, tol,
            ctypes.byref(call_d), ctypes.byref(call_b),
        )
        rel_err = abs(call_d.value - call_b.value) / (call_b.value + 1e-12)
        return {
            "passed":     bool(passed),
            "call_dirac": call_d.value,
            "call_bs":    call_b.value,
            "rel_err":    rel_err,
            "tolerance":  tol,
        }

    def version(self) -> str:
        buf = ctypes.create_string_buffer(256)
        self._lib.dirac_version(buf, 256)
        return buf.value.decode()

    def dhj_version(self) -> str:
        buf = ctypes.create_string_buffer(256)
        self._lib.dhj_version(buf, 256)
        return buf.value.decode()

    def predict_dhj(
        self,
        pair: str,
        spot: float,
        horizon_days: float  = 30.0,
        # Heston
        kappa_H: float       = 1.5,
        xi: float            = 0.30,
        rho: float           = -0.25,
        # Dirac
        kappa0: float        = 1.0,
        kappa1: float        = 0.002,
        delta_cp: float      = 0.0,
        # Jumps
        jump_lambda: float   = 5.0,
        jump_p_up: float     = 0.55,
        jump_eta_plus: float = 100.0,   # mean up-jump = 1% (FX calibrated)
        jump_eta_minus: float= 80.0,    # mean down-jump = 1.25%
        # MC
        n_paths: int         = 300,
        n_sites: int         = 400,
        seed: int            = 0,
    ) -> "DHJPrediction":
        """
        Run the Dirac-Heston-Jump hybrid model.

        Five layers over the base Dirac model:
          1. Heston stochastic variance  (kappa_H, xi, rho)
          2. Stochastic Dirac diffusion  D=(1-ρ²)v/2, c=√((1-ρ²)v)
          3. Regime-dependent mass       κ(v) = kappa0 + kappa1/v
          4. Leverage-correlated drift   ρ·√v·dW_v enters as path drift
          5. Kou double-exponential jumps (jump_lambda, jump_p_up, eta±)

        n_paths MC paths are averaged; ~300 gives good accuracy in <2s.
        """
        params = _PAIR_PARAMS.get(pair.upper(), {"sigma": 0.085, "r": 0.0500})
        sigma  = params["sigma"]
        r      = params["r"]
        v0     = sigma * sigma
        theta  = v0          # start at long-run variance = initial variance
        T      = horizon_days / 365.0
        n      = n_sites

        prices_arr  = (ctypes.c_double * n)()
        prob_dhj_arr= (ctypes.c_double * n)()
        prob_bs_arr = (ctypes.c_double * n)()
        scalars_arr = (ctypes.c_double * 10)()

        n_fill = self._lib.dhj_predict(
            spot, r, T,
            kappa_H, theta, xi, rho, v0,
            kappa0, kappa1, delta_cp,
            jump_lambda, jump_p_up, jump_eta_plus, jump_eta_minus,
            n, n_paths, ctypes.c_uint(seed),
            prices_arr, prob_dhj_arr, prob_bs_arr,
            n, scalars_arr,
        )

        if n_fill < 0:
            raise RuntimeError(f"dhj_predict failed for pair={pair}")

        return DHJPrediction(
            prices        = list(prices_arr[:n_fill]),
            prob_dhj      = list(prob_dhj_arr[:n_fill]),
            prob_bs       = list(prob_bs_arr[:n_fill]),
            mean_dhj      = scalars_arr[0],
            mean_bs       = scalars_arr[1],
            var_log_dhj   = scalars_arr[2],
            var_log_bs    = scalars_arr[3],
            call_dhj      = scalars_arr[4],
            call_bs       = scalars_arr[5],
            chiral_charge = scalars_arr[6],
            avg_variance  = scalars_arr[7],
            n_steps       = int(scalars_arr[8]),
            n_paths       = int(scalars_arr[9]),
            horizon_days  = horizon_days,
            kappa0        = kappa0,
            xi            = xi,
            rho           = rho,
            jump_lambda   = jump_lambda,
        )
