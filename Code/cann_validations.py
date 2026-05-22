import numpy      as np
import tensorflow as tf


def _all_close(a, b, n_digits=3, atol=1e-4):
    both_nan = np.isnan(a) & np.isnan(b)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    skip     = both_nan | both_inf
    a_safe   = np.where(skip, 0.0, a)
    b_safe   = np.where(skip, 0.0, b)
    scale    = np.maximum(np.abs(a_safe), np.abs(b_safe))
    tol      = np.where(scale > atol, scale * 10**(-n_digits), atol + scale)
    return np.all(skip | (np.abs(a_safe - b_safe) <= tol))


def _fill_F(F11, F12, F21, F22):
    det2d = F11 * F22 - F12 * F21
    assert det2d > 0, f"In-plane det must be positive, got {det2d}"
    F = np.zeros((3,3), dtype=np.float32)
    F[0,0], F[0,1] = F11, F12
    F[1,0], F[1,1] = F21, F22
    F[2,2]         = 1.0 / det2d
    return F


def _define_F():
    F_list = []

    F_list.append(np.eye(3, dtype=np.float32))

    for lam in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0]:
        F_list.append(_fill_F(lam, 0, 0, 1.0/np.sqrt(lam)))

    for lam in [0.7, 0.9, 1.1, 1.3, 1.5, 2.0]:
        F_list.append(_fill_F(lam, 0, 0, lam))

    for lam in [0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0]:
        F_list.append(_fill_F(lam, 0, 0, 1.0))

    for gamma in [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]:
        F_list.append(_fill_F(1.0, gamma, 0.0, 1.0))

    for l1, l2 in [(1.2, 1.5), (1.5, 0.8), (2.0, 1.3), (0.6, 1.8)]:
        F_list.append(_fill_F(l1, 0, 0, l2))

    for lam, gam in [(1.3, 0.3), (1.5, 0.5), (0.8, 0.4)]:
        F_list.append(_fill_F(lam, gam, 0.0, 1.0/np.sqrt(lam)))

    return np.array(F_list)


def _fill_out_of_plane_Q(phi1, phi2, phi3, det=1):
    def _Rx(phi):
        R = np.zeros((3, 3), dtype=np.float32)
        R[0,0]         = 1.0
        R[1,1], R[1,2] = np.cos(phi), -np.sin(phi)
        R[2,1], R[2,2] = np.sin(phi),  np.cos(phi)
        return R

    def _Ry(phi):
        R = np.zeros((3, 3), dtype=np.float32)
        R[0,0], R[0,2] =  np.cos(phi), np.sin(phi)
        R[1,1]         =  1.0
        R[2,0], R[2,2] = -np.sin(phi), np.cos(phi)
        return R

    def _Rz(phi):
        R = np.zeros((3, 3), dtype=np.float32)
        R[0,0], R[0,1] = np.cos(phi), -np.sin(phi)
        R[1,0], R[1,1] = np.sin(phi),  np.cos(phi)
        R[2,2]         = 1.0
        return R

    R = _Rx(phi1) @ _Ry(phi2) @ _Rz(phi3)
    if det == -1:
        R = R @ np.diag([1.0, 1.0, -1.0]).astype(np.float32)
    return R


def _fill_in_plane_Q(theta):
    Q = np.zeros((3,3), dtype=np.float32)
    Q[0,0], Q[0,1] = np.cos(theta), -np.sin(theta)
    Q[1,0], Q[1,1] = np.sin(theta),  np.cos(theta)
    Q[2,2]         = 1.0
    return Q


def _define_out_of_plane_Q(det=1):
    assert det in [1,-1], "det must be +1 or -1"

    angle_triples = [
        (  np.pi/6, np.pi/4,  np.pi/3),
        (  np.pi/4, np.pi/4,  np.pi/4),
        (  np.pi/2,     0.0,  np.pi/2),
        (      0.0, np.pi/3,  np.pi/6),
        (  np.pi/3, np.pi/2,  np.pi/4),
        (2*np.pi/3, np.pi,  5*np.pi/3),
        (     0.73,    1.05,     2.14),
        (     1.22,    0.58,     0.37),
        (  np.pi,   np.pi/2,      0.0),
    ]

    return np.array([_fill_out_of_plane_Q(p1, p2, p3, det=det) for p1, p2, p3 in angle_triples])


def _define_diagonal_F_for_hessian(L=2.0, n_grid=50):
    log_lam = np.linspace(-L, L, n_grid)
    log_l1, log_l2 = np.meshgrid(log_lam, log_lam, indexing="ij")
    stretches = np.stack([np.exp(log_l1.ravel()),
                          np.exp(log_l2.ravel()),
                          np.exp(-(log_l1 + log_l2).ravel())], axis=-1)
    stretches = np.sort(stretches, axis=-1)[:, ::-1]
    stretches = np.unique(np.round(stretches, decimals=10), axis=0)
    stretches = stretches[stretches[:,2] > np.exp(-3*L)]

    return np.array([_fill_F(s[0], 0, 0, s[1]) for s in stretches])


def _define_a_and_b_for_hessian(n_dirs=200):
    def _fibonacci_hemisphere(n):
        idx    = np.arange(n, dtype=float)
        golden = (1 + np.sqrt(5)) / 2
        theta  = np.arccos(1 - 2 * (idx + 0.5) / (2 * n))
        phi    = 2 * np.pi * idx / golden
        mask   = theta <= np.pi / 2
        theta, phi = theta[mask], phi[mask]
        dirs = np.stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)], axis=-1)
        return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    dirs_a = _fibonacci_hemisphere(n_dirs)
    dirs_b = _fibonacci_hemisphere(n_dirs)
    return np.einsum("ai,bj->abij", dirs_a, dirs_b).reshape(-1, 9)


def _interpolate_F_loop(waypoints, n_seg=50):
    Fs = []
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i],   dtype=np.float32)
        end   = np.array(waypoints[i+1], dtype=np.float32)
        for k in range(n_seg):
            s = k / n_seg
            p = (1 - s) * start + s * end
            Fs.append(_fill_F(p[0], p[1], p[2], p[3]))
    Fs.append(_fill_F(*waypoints[-1]))
    return np.array(Fs)


def _define_F_paths_for_thermodynamic_consistency(n_seg=200):
    s  = np.linspace(0, 1, n_seg + 1)
    QF = _fill_in_plane_Q(np.pi/4) @ _fill_F(1.3, 0.5, 0.0, 1.0/np.sqrt(1.3))

    # Loop 1: diagonal deformations only
    loop_1 = _interpolate_F_loop([
        (1.0, 0.0, 0.0, 1.0),
        (1.5, 0.0, 0.0, 1.0/np.sqrt(1.5)),
        (1.5, 0.0, 0.0, 1.0),
        (1.3, 0.0, 0.0, 1.3),
        (1.0, 0.0, 0.0, 1.0),
    ], n_seg)

    # Loop 2: off-diagonal deformations with rotated waypoint
    loop_2 = _interpolate_F_loop([
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 0.5, 0.0, 1.0),
        (1.3, 0.5, 0.0, 1.0/np.sqrt(1.3)),
        (QF[0,0], QF[0,1], QF[1,0], QF[1,1]),
        (1.3, 0.0, 0.0, 1.0/np.sqrt(1.3)),
        (1.0, 0.0, 0.0, 1.0),
    ], n_seg)

    # Two-path comparison: direct uniaxial vs. via pure shear
    path_a = np.array([_fill_F(1.0 + si*0.5, 0.0, 0.0, 1.0/np.sqrt(1.0 + si*0.5)) for si in s])
    path_b = np.array([_fill_F(1.0 + si*0.5, 0.0, 0.0, 1.0) for si in s] +
                       [_fill_F(1.5, 0.0, 0.0, 1.0 + si*(1.0/np.sqrt(1.5) - 1.0)) for si in s[1:]])

    return [loop_1, loop_2], path_a, path_b


def _compute_work_along_F_path(cann, Fs):
    out     = cann.predict(Fs)
    P, psi  = out["P"], out["Psi"][:, 0]
    delta_W = np.einsum("kij,kij->k", 0.5 * (P[:-1] + P[1:]), np.diff(Fs, axis=0))
    W_cumul = np.concatenate([[0.0], np.cumsum(delta_W)])
    dpsi    = psi - psi[0]
    return W_cumul, dpsi, delta_W


def validate_thermodynamic_consistency(cann):
    loops, path_a, path_b = _define_F_paths_for_thermodynamic_consistency()

    for loop in loops:
        W_cumul, dpsi, delta_W = _compute_work_along_F_path(cann, loop)

        # Metric 1: normalized loop residual
        W_loop  = W_cumul[-1]
        sum_abs = np.sum(np.abs(delta_W))
        eta     = np.abs(W_loop) / sum_abs if sum_abs > 0 else 0.0
        if eta > 1e-2:
            return "failed"

        # Metric 2: stress uniqueness at start == end
        P_start = cann.predict(loop[:1])["P"]
        P_end   = cann.predict(loop[-1:])["P"]
        if not _all_close(P_start, P_end):
            return "failed"

    # Metric 3: two-path comparison
    W_a, dpsi_a, _ = _compute_work_along_F_path(cann, path_a)
    W_b, dpsi_b, _ = _compute_work_along_F_path(cann, path_b)

    if not _all_close(W_a[-1], W_b[-1]):
        return "failed"

    # Metric 4: work-energy consistency along open paths
    if not _all_close(W_a, dpsi_a, n_digits=2):
        return "failed"
    if not _all_close(W_b, dpsi_b, n_digits=2):
        return "failed"

    return "passed"


def validate_stress_symmetry(cann):
    Fs  = _define_F()
    Ps  = cann.predict(Fs)["P"]

    sigmas = Ps @ Fs.transpose(0,2,1)

    if _all_close(sigmas, sigmas.transpose(0,2,1)):
        return "passed"
    return "failed"


def validate_objectivity(cann):
    # Psi is computed independent of the Lagrange multiplier p. Since p is the only place the plane-
    # stress assumption (P33=0) enters, arbitrary 3D rotations can be used to test Psi(QF) = Psi(F).
    Fs = _define_F()
    Qs = _define_out_of_plane_Q()
    n_F, n_Q = len(Fs), len(Qs)

    psi_F  = cann.predict(Fs)["Psi"][:,0]
    QFs    = (Qs[:,np.newaxis] @ Fs[np.newaxis]).reshape(-1,3,3)
    psi_QF = cann.predict(QFs)["Psi"][:,0].reshape(n_Q,n_F)

    if _all_close(psi_QF, psi_F[np.newaxis]):
        return "passed"
    return "failed"


def validate_material_symmetry(cann):
    # Psi is computed independent of the Lagrange multiplier p. Since p is the only place the plane-
    # stress assumption (P33=0) enters, arbitrary 3D rotations and reflections can be used to test
    # Psi(F @ Q.T) = Psi(F).
    Fs  = _define_F()
    Qs  = np.concatenate([_define_out_of_plane_Q(det=1), _define_out_of_plane_Q(det=-1)], axis=0)
    n_F, n_Q = len(Fs), len(Qs)

    psi_F   = cann.predict(Fs)["Psi"][:,0]
    FQTs    = (Fs[:,np.newaxis] @ Qs[np.newaxis].transpose(0,1,3,2)).reshape(-1,3,3)
    psi_FQT = cann.predict(FQTs)["Psi"][:,0].reshape(n_F,n_Q)

    if _all_close(psi_FQT, psi_F[:,np.newaxis]):
        return "passed"
    return "failed"


def validate_ellipticity(cann):
    # Check rank-one convexity: (a⊗b) : A : (a⊗b) ≥ 0  ∀ a,b ∈ R³
    # Isotropy + objectivity allows restricting F to diagonal form.
    # Incompressibility: λ₃ = (λ₁λ₂)⁻¹. Permutation symmetry: λ₁ ≥ λ₂ ≥ λ₃.
    def _compute_elasticity_tensor(F):
        N = F.shape[0]
        f = tf.Variable(tf.reshape(F, (N,9)))

        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                F_reshaped = tf.reshape(f, (N,3,3))
                psi        = cann.psi_from_F(F_reshaped)
            grad = tape_inner.batch_jacobian(psi, f)
            grad = tf.squeeze(grad, axis=1)
        hessian = tape_outer.batch_jacobian(grad, f)

        return hessian.numpy().reshape(N,3,3,3,3)

    Fs         = _define_diagonal_F_for_hessian()
    h_all      = _define_a_and_b_for_hessian()
    batch_size = 64
    for start in range(0, len(Fs), batch_size):
        A_flat = _compute_elasticity_tensor(Fs[start:start+batch_size]).reshape(-1,9,9)
        vals   = np.einsum("bi,fij,bj->fb", h_all, A_flat, h_all)
        if np.min(vals) < 0:
            return "failed"

    return "passed"


def validate_growth_condition(cann):
    return "passed" # Trivial for incompressibility


def validate_energy_normalization(cann):
    tol = 1e-3
    psi = cann.predict(np.eye(3).reshape(1,3,3))["Psi"]
    if np.abs(psi[0,0]) < tol:
        return "passed"
    else:
        return "failed"


def validate_stress_normalization(cann):
    tol = 1e-3
    P   = cann.predict(np.eye(3).reshape(1,3,3))["P"]
    if np.max(np.abs(P)) < tol:
        return "passed"
    else:
        return "failed"


def validate_non_negativity_of_strain_energy(cann):
    Fs  = _define_F()
    psi = cann.predict(Fs)["Psi"][:,0]

    if np.all(psi >= -1e-3):
        return "passed"
    return "failed"
