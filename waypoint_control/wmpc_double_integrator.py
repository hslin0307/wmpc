# file: waypoint_control/wmpc_double_integrator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import cvxpy as cp


@dataclass
class JointLimits:
    q_min: np.ndarray        # (m,)
    q_max: np.ndarray        # (m,)
    qd_min: np.ndarray       # (m,)
    qd_max: np.ndarray       # (m,)
    qdd_min: np.ndarray      # (m,)
    qdd_max: np.ndarray      # (m,)


@dataclass
class WMPCConfig:
    dof: int                 # m
    h: float                 # sampling time
    N_max: int               # maximum horizon length
    eps: float               # waypoint / goal tolerance
    gamma: float             # smooth cost parameter (同論文 γ)
    sigma: float             # weight scaling
    d_min: float             # distance lower bound for weights
    N_min_goal: int = 5      # minimal horizon for goal
    w_input_reg: float = 1e-3
    w_collision: float = 0.0 # set >0 when you implement collision cost


def fo_discretization_double_integrator(dof: int, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    First-order hold discretization for double integrator:
        x = [q, qd], u = qdd
        q_{k+1}   = q_k + h qd_k + h^2/3 u_k + h^2/6 u_{k+1}
        qd_{k+1}  = qd_k + h/2 u_k + h/2 u_{k+1}
    """
    I = np.eye(dof)
    zero = np.zeros_like(I)

    Phi = np.block([[I, h * I],
                    [zero, I]])

    Gamma1 = np.vstack([
        (h ** 2 / 3.0) * I,
        (h / 2.0) * I
    ])

    Gamma2 = np.vstack([
        (h ** 2 / 6.0) * I,
        (h / 2.0) * I
    ])

    return Phi, Gamma1, Gamma2


def smooth_l1(q: cp.Expression, q_target: np.ndarray, gamma: float) -> cp.Expression:
    """
    Smooth 1-norm exactly matching paper:
        sum_i ( sqrt((q_i - q*_i)^2 + gamma^2) - gamma )

    用 DCP 合法的方式表示：
        sqrt(e^2 + gamma^2) = || [e, gamma] ||_2
    """
    diff = q - q_target  # shape (m,)
    m = q_target.shape[0]
    terms: List[cp.Expression] = []
    for i in range(m):
        ei = diff[i]
        # ||[ei, gamma]||_2 - gamma  等價於 sqrt(ei^2 + gamma^2) - gamma
        terms.append(cp.norm(cp.hstack([ei, gamma])) - gamma)
    return cp.sum(terms)


def compute_cost_weights(
    q_init: np.ndarray, q_w: np.ndarray, q_g: np.ndarray, sigma: float, d_min: float
) -> Tuple[float, float]:
    """
    w1, w2 weights as in equations (9) and (10):
        w1 = sigma / max( ||q_w - q_init||_2, d_min )
        w2 = sigma / max( ||q_g - q_w||_2, d_min )
    """
    d1 = max(float(np.linalg.norm(q_w - q_init)), d_min)
    d2 = max(float(np.linalg.norm(q_g - q_w)), d_min)
    return sigma / d1, sigma / d2


def check_goal_reachability(
    q_traj: np.ndarray,
    q_target: np.ndarray,
    eps: float,
    N_start: int,
    N_stop: int,
) -> int:
    """
    Approximation of Algorithm 2 (Check Goal Reachability) for joint space.
    q_traj: shape (N, m), sequence of joint positions along horizon
    q_target: shape (m,), target joint configuration
    eps: tolerance band
    Returns:
        index i in [N_start+1, N_stop-1] where goal becomes reachable (first time),
        or N_stop if not reachable in that interval.
    """
    N, m = q_traj.shape
    reached = np.zeros(m, dtype=bool)

    for i in range(max(N_start + 1, 1), min(N_stop, N)):
        qi = q_traj[i]
        qim1 = q_traj[i - 1]

        for j in range(m):
            if abs(qi[j] - q_target[j]) <= eps:
                reached[j] = True
            else:
                s1 = np.sign(qim1[j] - q_target[j])
                s2 = np.sign(qi[j] - q_target[j])
                if s1 != 0.0 and s2 != 0.0 and s1 != s2:
                    reached[j] = True

        if np.all(reached):
            return i

    return N_stop


class WaypointMPC:
    """
    Double-integrator wMPC (joint space) with FOH.
    """

    def __init__(self, cfg: WMPCConfig, limits: JointLimits):
        self.cfg = cfg
        self.limits = limits
        self.Phi, self.Gamma1, self.Gamma2 = fo_discretization_double_integrator(cfg.dof, cfg.h)

        self.Ns = cfg.N_max
        self.N = cfg.N_max

        self._x_prev: Optional[np.ndarray] = None  # (N, 2m)
        self._u_prev: Optional[np.ndarray] = None  # (N, m)

    @property
    def dof(self) -> int:
        return self.cfg.dof

    @property
    def state_dim(self) -> int:
        return 2 * self.cfg.dof

    def _build_warm_start(
        self, N: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        m = self.dof
        nx = 2 * m

        if self._x_prev is None or self._u_prev is None:
            x_ws = np.zeros((N, nx))
            u_ws = np.zeros((N, m))
            return x_ws, u_ws

        x_prev = self._x_prev
        u_prev = self._u_prev
        N_prev = x_prev.shape[0]

        x_ws = np.zeros((N, nx))
        u_ws = np.zeros((N, m))

        if N <= N_prev - 1:
            x_ws[:] = x_prev[1:1 + N]
            u_ws[:] = u_prev[1:1 + N]
        else:
            count = N_prev - 1
            if count > 0:
                x_ws[:count] = x_prev[1:]
                u_ws[:count] = u_prev[1:]
                x_ws[count:] = x_prev[-1]
                u_ws[count:] = u_prev[-1]
            else:
                x_ws[:] = x_prev[-1]
                u_ws[:] = u_prev[-1]

        return x_ws, u_ws

    def _update_horizons(self, q_pred: np.ndarray, q_w: np.ndarray, q_g: np.ndarray) -> None:
        cfg = self.cfg
        Nmax = cfg.N_max

        i_w = check_goal_reachability(
            q_traj=q_pred,
            q_target=q_w,
            eps=cfg.eps,
            N_start=0,
            N_stop=Nmax,
        )
        if i_w < Nmax:
            self.Ns = max(0, i_w - 1)

        i_g = check_goal_reachability(
            q_traj=q_pred,
            q_target=q_g,
            eps=cfg.eps,
            N_start=self.Ns,
            N_stop=Nmax,
        )
        if i_g < Nmax:
            self.N = max(cfg.N_min_goal, i_g - 1)

        self.Ns = int(np.clip(self.Ns, 0, cfg.N_max))
        self.N = int(np.clip(self.N, cfg.N_min_goal, cfg.N_max))

    def solve(
        self,
        q_meas: np.ndarray,
        qd_meas: np.ndarray,
        q_w: np.ndarray,
        q_g: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve one wMPC step.
        Returns:
            u0: (m,) first optimal acceleration
            x_opt: (N, 2m)
            u_opt: (N, m)
        """
        cfg = self.cfg
        m = self.dof
        nx = 2 * m

        q_meas = np.asarray(q_meas, dtype=float).reshape(m)
        qd_meas = np.asarray(qd_meas, dtype=float).reshape(m)
        q_w = np.asarray(q_w, dtype=float).reshape(m)
        q_g = np.asarray(q_g, dtype=float).reshape(m)

        x0 = np.concatenate([q_meas, qd_meas])

        # simple predicted trajectory for horizon update: zero-input constant-velocity
        q_pred_simple = np.zeros((cfg.N_max, m))
        q_pred_simple[0] = q_meas.copy()
        qd_tmp = qd_meas.copy()
        q_tmp = q_meas.copy()
        for k in range(1, cfg.N_max):
            q_tmp = q_tmp + cfg.h * qd_tmp
            q_pred_simple[k] = q_tmp

        self._update_horizons(q_pred_simple, q_w, q_g)
        Ns = self.Ns
        N = self.N

        w1, w2 = compute_cost_weights(q_meas, q_w, q_g, cfg.sigma, cfg.d_min)

        x_ws, u_ws = self._build_warm_start(N)

        x = cp.Variable((N, nx))
        u = cp.Variable((N, m))

        constraints: List[cp.Constraint] = []

        constraints.append(x[0, :] == x0)

        for k in range(N - 1):
            xk = x[k, :]
            uk = u[k, :]
            ukp1 = u[k + 1, :]
            xkp1 = x[k + 1, :]
            constraints.append(
                xkp1 == self.Phi @ xk + self.Gamma1 @ uk + self.Gamma2 @ ukp1
            )

        qd_terminal = x[N - 1, m:]
        constraints.append(qd_terminal == 0)
        constraints.append(u[N - 1, :] == 0)

        lim = self.limits
        q_min = lim.q_min.reshape(m)
        q_max = lim.q_max.reshape(m)
        qd_min = lim.qd_min.reshape(m)
        qd_max = lim.qd_max.reshape(m)
        qdd_min = lim.qdd_min.reshape(m)
        qdd_max = lim.qdd_max.reshape(m)

        for k in range(N):
            qk = x[k, :m]
            qdk = x[k, m:]
            uk = u[k, :]

            constraints += [
                qk >= q_min,
                qk <= q_max,
                qdk >= qd_min,
                qdk <= qd_max,
                uk >= qdd_min,
                uk <= qdd_max,
            ]

        if 0 <= Ns < N - 1:
            q_w_step = x[Ns - 1, :m]
            constraints += [
                q_w_step >= (q_w - cfg.eps),
                q_w_step <= (q_w + cfg.eps),
            ]

        if N < cfg.N_max:
            q_goal_step = x[N - 1, :m]
            constraints += [
                q_goal_step >= (q_g - cfg.eps),
                q_goal_step <= (q_g + cfg.eps),
            ]

        obj_terms: List[cp.Expression] = []

        for k in range(N):
            qk = x[k, :m]
            if k < Ns:
                obj_terms.append(w1 * smooth_l1(qk, q_w, cfg.gamma))
            else:
                obj_terms.append(w2 * smooth_l1(qk, q_g, cfg.gamma))

        for k in range(N):
            obj_terms.append(cfg.w_input_reg * cp.sum_squares(u[k, :]))

        # TODO: collision cost hook

        objective = cp.Minimize(cp.sum(obj_terms))
        problem = cp.Problem(objective, constraints)

        x.value = x_ws
        u.value = u_ws

        problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"wMPC solve failed, status={problem.status}")

        x_opt = x.value
        u_opt = u.value

        self._x_prev = x_opt.copy()
        self._u_prev = u_opt.copy()

        u0 = u_opt[0, :].copy()
        return u0, x_opt, u_opt


if __name__ == "__main__":
    # quick sanity test
    m = 2
    h = 0.1

    q_min = np.array([-2.0, -2.0])
    q_max = np.array([2.0, 2.0])
    qd_min = -np.ones(m) * 2.0
    qd_max = np.ones(m) * 2.0
    qdd_min = -np.ones(m) * 5.0
    qdd_max = np.ones(m) * 5.0

    limits = JointLimits(q_min, q_max, qd_min, qd_max, qdd_min, qdd_max)

    cfg = WMPCConfig(
        dof=m,
        h=h,
        N_max=15,
        eps=0.02,
        gamma=0.1,
        sigma=10.0,
        d_min=0.05,
        N_min_goal=5,
        w_input_reg=1e-3,
    )

    wmpc = WaypointMPC(cfg, limits)
    q0 = np.array([0.0, 0.0])
    qd0 = np.array([0.0, 0.0])
    q_w = np.array([0.5, 0.0])
    q_g = np.array([1.0, 0.5])

    u0, x_opt, u_opt = wmpc.solve(q0, qd0, q_w, q_g)
    print("u0:", u0)
    print("qN:", x_opt[-1, :m])
