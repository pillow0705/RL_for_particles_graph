import numpy as np


def pbc_diff(p1, p2, L):
    d = p1 - p2
    return d - np.round(d / L) * L


def solve_three_spheres(p1, r1, p2, r2, p3, r3, r_new):
    dp21 = p2 - p1
    d12  = np.linalg.norm(dp21)
    s1, s2, s3 = r1 + r_new, r2 + r_new, r3 + r_new

    if d12 > s1 + s2 or d12 < abs(s1 - s2):
        return False, None, None

    ex   = dp21 / d12
    dp31 = p3 - p1
    i    = np.dot(ex, dp31)
    ey_v = dp31 - i * ex
    d_ey = np.linalg.norm(ey_v)

    if d_ey < 1e-7:
        return False, None, None
    ey = ey_v / d_ey
    ez = np.cross(ex, ey)

    x    = (s1**2 - s2**2 + d12**2) / (2 * d12)
    y    = (s1**2 - s3**2 + i**2 + d_ey**2) / (2 * d_ey) - (i * x) / d_ey
    z_sq = s1**2 - x**2 - y**2

    if z_sq < 0:
        return False, None, None
    z = np.sqrt(z_sq)

    sol1 = p1 + x * ex + y * ey + z * ez
    sol2 = p1 + x * ex + y * ey - z * ez
    return True, sol1, sol2


def check_collision(sol, r_new, all_pos, all_rad, L, tol):
    """检查候选点与所有已有粒子是否碰撞。返回 (collision, coordination)。"""
    collision    = False
    coordination = 0
    for m in range(len(all_pos)):
        r_sum = all_rad[m] + r_new
        gap   = np.linalg.norm(pbc_diff(all_pos[m], sol, L)) - r_sum
        if gap < -tol * r_sum:
            return True, 0
        if gap < tol * r_sum:
            coordination += 1
    return collision, coordination


def check_single_collision(sol, r_new, new_pos, new_rad, L, tol):
    """检查候选点与单个新粒子是否碰撞（增量过滤用）。返回 (collision, touching)。"""
    r_sum = new_rad + r_new
    gap   = np.linalg.norm(pbc_diff(new_pos, sol, L)) - r_sum
    return gap < -tol * r_sum, gap < tol * r_sum
