from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from scipy.stats import t as student_t


@dataclass
class OLSResult:
    slope: float
    intercept: float
    p_value: float


def ols_slope_p(x: np.ndarray, y: np.ndarray) -> OLSResult:
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return OLSResult(np.nan, np.nan, np.nan)
    xm = x.mean(); ym = y.mean()
    Sxx = np.sum((x - xm) ** 2)
    if Sxx == 0:
        return OLSResult(np.nan, np.nan, np.nan)
    Sxy = np.sum((x - xm) * (y - ym))
    slope = Sxy / Sxx
    intercept = ym - slope * xm
    yhat = intercept + slope * x
    resid = y - yhat
    dof = max(x.size - 2, 1)
    sigma2 = np.sum(resid ** 2) / dof
    se_slope = np.sqrt(sigma2 / Sxx)
    t_stat = slope / se_slope if se_slope > 0 else np.nan
    p_val = 2 * (1 - student_t.cdf(abs(t_stat), df=dof)) if np.isfinite(t_stat) else np.nan
    return OLSResult(float(slope), float(intercept), float(p_val))


def bin_mean_sem(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"x": x, "y": y})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        centers = (bins[:-1] + bins[1:]) / 2
        return centers, np.full_like(centers, np.nan, dtype=float), np.full_like(centers, np.nan, dtype=float)
    df["bin"] = pd.cut(df["x"], bins, include_lowest=True)
    grouped = df.groupby("bin", observed=False)["y"]
    means = grouped.mean().to_numpy()
    sems = grouped.sem().to_numpy()
    centers = (bins[:-1] + bins[1:]) / 2
    # Align lengths if grouping missed empty bins
    if means.size != centers.size:
        idx = pd.cut(centers, bins, include_lowest=True)
        means = grouped.mean().reindex(idx).to_numpy()
        sems = grouped.sem().reindex(idx).to_numpy()
    return centers, means, sems


@dataclass
class CosineFit:
    a: float
    b: float
    c: float
    p_a: float
    p_b: float
    converged: bool


def fit_cosine_series_deg(x_deg: np.ndarray, y: np.ndarray) -> CosineFit:
    """Fit y ~ a*cos(theta)+b*cos(2*theta)+c using least squares; return params and p-values for a and b."""
    x = np.asarray(x_deg)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        mu = float(np.nanmean(y)) if y.size else 0.0
        return CosineFit(np.nan, np.nan, mu, np.nan, np.nan, False)
    theta = np.radians(x)
    X = np.column_stack([np.cos(theta), np.cos(2 * theta), np.ones_like(theta)])
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = [float(v) for v in beta]
        n, p = X.shape
        dof = max(n - p, 1)
        if residuals.size == 0:
            # perfect fit; fallback to nan p-values
            return CosineFit(a, b, c, np.nan, np.nan, True)
        sigma2 = float(residuals[0]) / dof
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        se_a = float(se[0]); se_b = float(se[1])
        t_a = (a / se_a) if se_a > 0 else np.nan
        t_b = (b / se_b) if se_b > 0 else np.nan
        p_a = 2 * (1 - student_t.cdf(abs(t_a), df=dof)) if np.isfinite(t_a) else np.nan
        p_b = 2 * (1 - student_t.cdf(abs(t_b), df=dof)) if np.isfinite(t_b) else np.nan
        return CosineFit(a, b, c, float(p_a), float(p_b), True)
    except Exception:
        mu = float(np.nanmean(y)) if y.size else 0.0
        return CosineFit(np.nan, np.nan, mu, np.nan, np.nan, False)


@dataclass
class LegendreFit:
    coeffs: np.ndarray
    order: int
    success: bool
    r2: float


def legendre_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int = 3,
    x_min: float = -1.0,
    x_max: float = 1.0,
) -> LegendreFit:
    """Least-squares Legendre polynomial fit (default up to 3rd order).

    The input x range is scaled to [-1, 1] before constructing the Vandermonde matrix
    to keep the basis well conditioned.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return LegendreFit(np.full(order + 1, np.nan), order, False, np.nan)
    x = x[mask]
    y = y[mask]
    if x.size < order + 2 or x_max == x_min:
        return LegendreFit(np.full(order + 1, np.nan), order, False, np.nan)

    # Scale to [-1, 1] for Legendre basis
    scale = 2.0 / (x_max - x_min)
    x_scaled = (x - x_min) * scale - 1.0

    try:
        X = np.polynomial.legendre.legvander(x_scaled, order)
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        y_hat = np.polynomial.legendre.legval(x_scaled, coeffs)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return LegendreFit(coeffs.astype(float), order, True, r2)
    except Exception:
        return LegendreFit(np.full(order + 1, np.nan), order, False, np.nan)


def eval_legendre(x: np.ndarray, fit: LegendreFit, *, x_min: float = -1.0, x_max: float = 1.0) -> np.ndarray:
    """Evaluate a LegendreFit at arbitrary x values."""
    if not fit.success or fit.coeffs.size == 0 or x_max == x_min:
        return np.full_like(x, np.nan, dtype=float)
    x = np.asarray(x)
    scale = 2.0 / (x_max - x_min)
    x_scaled = (x - x_min) * scale - 1.0
    return np.polynomial.legendre.legval(x_scaled, fit.coeffs)


@dataclass
class PiecewiseLinearFit:
    intercept: float
    slope_neg: float
    slope_pos: float
    success: bool
    r2: float


def piecewise_linear_shared_intercept(x: np.ndarray, y: np.ndarray) -> PiecewiseLinearFit:
    """Fit two-line model sharing an intercept at x=0 (segments: x<=0 and x>=0)."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return PiecewiseLinearFit(np.nan, np.nan, np.nan, False, np.nan)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return PiecewiseLinearFit(np.nan, np.nan, np.nan, False, np.nan)

    x_neg = np.minimum(x, 0.0)
    x_pos = np.maximum(x, 0.0)
    X = np.column_stack([np.ones_like(x), x_neg, x_pos])
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        intercept, slope_neg, slope_pos = [float(v) for v in beta]
        y_hat = intercept + slope_neg * x_neg + slope_pos * x_pos
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return PiecewiseLinearFit(intercept, slope_neg, slope_pos, True, r2)
    except Exception:
        return PiecewiseLinearFit(np.nan, np.nan, np.nan, False, np.nan)


def eval_piecewise_linear(x: np.ndarray, fit: PiecewiseLinearFit) -> np.ndarray:
    """Evaluate a PiecewiseLinearFit at arbitrary x values."""
    if not fit.success:
        return np.full_like(x, np.nan, dtype=float)
    x = np.asarray(x)
    x_neg = np.minimum(x, 0.0)
    x_pos = np.maximum(x, 0.0)
    return fit.intercept + fit.slope_neg * x_neg + fit.slope_pos * x_pos
