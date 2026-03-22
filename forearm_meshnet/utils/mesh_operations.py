import numpy as np
from typing import Tuple, Optional
from scipy.spatial import cKDTree


#]Helpers: closed-form alignments 

def _kabsch_rigid(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Best-fit rigid transform (no scale) mapping A -> B
    using the Kabsch algorithm.
    Returns (R, t) such that B ≈ R @ A + t.
    """
    # A, B: (N,3)
    assert A.shape == B.shape and A.shape[1] == 3
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    Ac = A - muA
    Bc = B - muB

    H = Ac.T @ Bc          # (3,3)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Enforce proper rotation (det=+1) — avoid reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = muB - R @ muA
    return R, t


def _umeyama_similarity(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Best-fit similarity transform (rotation, translation, isotropic scale)
    mapping A -> B using Umeyama (1991).
    Returns (R, t, s) such that B ≈ s * (R @ A) + t.
    """
    # A, B: (N,3)
    assert A.shape == B.shape and A.shape[1] == 3
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    Ac = A - muA
    Bc = B - muB

    cov = (Bc.T @ Ac) / A.shape[0]  # note order for Umeyama
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    varA = (Ac ** 2).sum() / A.shape[0]
    s = float(D @ np.diag(S)) / max(varA, 1e-12)
    t = muB - s * (R @ muA)
    return R, t, float(s)


# ICP: iterative closest point (rigid & similarity) 

def rigid_icp(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    max_iters: int = 20,
    tol: float = 1e-4,
    init_R: Optional[np.ndarray] = None,
    init_t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rigid ICP (no scale). Finds (R, t) such that target ≈ R @ source + t.

    Args:
        source_pts: (Ns,3) source points
        target_pts: (Nt,3) target points
        max_iters: maximum ICP iterations
        tol: relative MSE improvement threshold to early-stop
        init_R: optional (3,3) initial rotation
        init_t: optional (3,) initial translation

    Returns:
        (R, t) cumulative rotation and translation (mapping original source to target frame)
    """
    assert source_pts.ndim == 2 and source_pts.shape[1] == 3
    assert target_pts.ndim == 2 and target_pts.shape[1] == 3
    src = source_pts.astype(np.float64).copy()
    dst = target_pts.astype(np.float64)

    # KD-Tree on target (fixed)
    kdt = cKDTree(dst)

    # Initialize cumulative transform
    R_tot = np.eye(3, dtype=np.float64) if init_R is None else init_R.astype(np.float64)
    t_tot = np.zeros(3, dtype=np.float64) if init_t is None else init_t.astype(np.float64)

    # Current transformed source
    cur = (src @ R_tot.T) + t_tot

    prev_err = np.inf
    for _ in range(max_iters):
        dists, idx = kdt.query(cur, k=1)
        matched = dst[idx]  # (Ns,3)

        # Estimate incremental rigid transform cur -> matched
        dR, dt = _kabsch_rigid(cur, matched)

        # Update cumulative transform: new = dR @ (R_tot X + t_tot) + dt
        R_tot = dR @ R_tot
        t_tot = dR @ t_tot + dt

        # Apply to source to get new 'cur'
        cur = (src @ R_tot.T) + t_tot

        mse = float((np.square(cur - matched)).mean())
        if prev_err - mse <= max(tol * prev_err, 1e-12):
            break
        prev_err = mse

    return R_tot, t_tot


def similarity_icp(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-5,
    init_R: Optional[np.ndarray] = None,
    init_t: Optional[np.ndarray] = None,
    init_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Similarity ICP (with isotropic scale). Finds (R, t, s) such that target ≈ s * (R @ source) + t.

    Args:
        source_pts: (Ns,3) source points
        target_pts: (Nt,3) target points
        max_iters: maximum ICP iterations
        tol: relative MSE improvement threshold to early-stop
        init_R: optional (3,3) initial rotation
        init_t: optional (3,) initial translation
        init_s: optional initial scale

    Returns:
        (R, t, s) cumulative transform mapping original source to target frame
    """
    assert source_pts.ndim == 2 and source_pts.shape[1] == 3
    assert target_pts.ndim == 2 and target_pts.shape[1] == 3
    src = source_pts.astype(np.float64).copy()
    dst = target_pts.astype(np.float64)

    # KD-Tree on target (fixed)
    kdt = cKDTree(dst)

    # Initialize cumulative similarity transform
    R_tot = np.eye(3, dtype=np.float64) if init_R is None else init_R.astype(np.float64)
    t_tot = np.zeros(3, dtype=np.float64) if init_t is None else init_t.astype(np.float64)
    s_tot = 1.0 if init_s is None else float(init_s)

    # Current transformed source: cur = s * (R @ src) + t
    cur = (s_tot * (src @ R_tot.T)) + t_tot

    prev_err = np.inf
    for _ in range(max_iters):
        dists, idx = kdt.query(cur, k=1)
        matched = dst[idx]  # (Ns,3)

        # Estimate incremental similarity transform cur -> matched
        dR, dt, ds = _umeyama_similarity(cur, matched)

        # Compose: new = ds * (dR @ cur) + dt
        # cur = s_tot * (R_tot @ src) + t_tot
        # => overall: s' = ds*s_tot; R' = dR @ R_tot; t' = ds*(dR @ t_tot) + dt
        R_tot = dR @ R_tot
        t_tot = (ds * (dR @ t_tot)) + dt
        s_tot = ds * s_tot

        # Update cur
        cur = (s_tot * (src @ R_tot.T)) + t_tot

        mse = float((np.square(cur - matched)).mean())
        if prev_err - mse <= max(tol * prev_err, 1e-12):
            break
        prev_err = mse

    return R_tot, t_tot, float(s_tot)


# Public aliases for the closed-form alignment helpers
kabsch_align = _kabsch_rigid
umeyama_align = _umeyama_similarity
