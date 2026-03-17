import numpy as np
from numba import njit


@njit(inline='always')
def pbc_diff_njit(p1, p2, L):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return np.array([
        dx - round(dx / L) * L,
        dy - round(dy / L) * L,
        dz - round(dz / L) * L
    ])


@njit
def solve_three_spheres_njit(p1, r1, p2, r2, p3, r3, r_new):
    dp21 = p2 - p1
    d12  = np.sqrt(np.sum(dp21 ** 2))
    s1, s2, s3 = r1 + r_new, r2 + r_new, r3 + r_new

    if d12 > s1 + s2 or d12 < abs(s1 - s2):
        return False, np.zeros(3), np.zeros(3)

    ex   = dp21 / d12
    dp31 = p3 - p1
    i    = np.dot(ex, dp31)
    ey_v = dp31 - i * ex
    d_ey = np.sqrt(np.sum(ey_v ** 2))

    if d_ey < 1e-7:
        return False, np.zeros(3), np.zeros(3)
    ey = ey_v / d_ey

    ez = np.array([
        ex[1] * ey[2] - ex[2] * ey[1],
        ex[2] * ey[0] - ex[0] * ey[2],
        ex[0] * ey[1] - ex[1] * ey[0]
    ])

    x    = (s1 ** 2 - s2 ** 2 + d12 ** 2) / (2 * d12)
    y    = (s1 ** 2 - s3 ** 2 + i ** 2 + d_ey ** 2) / (2 * d_ey) - (i * x) / d_ey
    z_sq = s1 ** 2 - x ** 2 - y ** 2

    if z_sq < 0:
        return False, np.zeros(3), np.zeros(3)
    z = np.sqrt(z_sq)

    sol1 = p1 + x * ex + y * ey + z * ez
    sol2 = p1 + x * ex + y * ey - z * ez
    return True, sol1, sol2


@njit
def check_collision_njit(sol, r_new, all_pos, all_rad, L, collision_tol):
    """检查候选点 sol 与所有已有粒子是否碰撞。返回 (collision, coordination)。"""
    collision    = False
    coordination = 0
    for m in range(len(all_pos)):
        dm   = pbc_diff_njit(all_pos[m], sol, L)
        dist = np.sqrt(np.sum(dm ** 2))
        gap  = dist - (all_rad[m] + r_new)
        if gap < -collision_tol:
            collision = True
            break
        if gap < collision_tol:
            coordination += 1
    return collision, coordination


@njit
def check_single_collision_njit(sol, r_new, new_pos, new_rad, L, collision_tol):
    """只检查候选点 sol 与单个新粒子是否碰撞（增量过滤用）。返回 (collision, touching)。"""
    dm        = pbc_diff_njit(new_pos, sol, L)
    dist      = np.sqrt(np.sum(dm ** 2))
    gap       = dist - (new_rad + r_new)
    collision = gap < -collision_tol
    touching  = gap < collision_tol
    return collision, touching


def get_pbc_center_of_mass(pos_array, L):
    theta   = 2 * np.pi * pos_array / L
    cos_sum = np.mean(np.cos(theta), axis=0)
    sin_sum = np.mean(np.sin(theta), axis=0)
    phi     = np.arctan2(sin_sum, cos_sum)
    phi     = np.where(phi < 0, phi + 2 * np.pi, phi)
    return (phi / (2 * np.pi)) * L
