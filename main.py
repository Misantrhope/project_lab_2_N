import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def accelerations_direct(pos, mass, G=1.0, eps=0.05, L=None):
    N, D = pos.shape
    diff = pos[None, :, :] - pos[:, None, :]
    if L is not None:
        diff = diff - L * np.round(diff / L)
    dist2 = np.sum(diff ** 2, axis=-1) + eps ** 2
    np.fill_diagonal(dist2, np.inf)
    inv_r3 = 1.0 / (dist2 * np.sqrt(dist2))
    acc = G * np.sum(diff * (mass[None, :, None] * inv_r3[..., None]),
                     axis=1)
    return acc


class OctreeNode:
    __slots__ = ("center", "size", "mass", "com",
                 "children", "indices", "is_leaf")

    def __init__(self, center, size):
        self.center = np.asarray(center, dtype=float)
        self.size = float(size)
        self.mass = 0.0
        self.com = np.zeros_like(self.center)
        self.children = []
        self.indices = []
        self.is_leaf = True


def build_tree(pos, mass, center, size, indices, max_per_leaf=1):
    node = OctreeNode(center, size)
    if len(indices) == 0:
        return node
    m = mass[indices]
    node.mass = float(m.sum())
    node.com = (pos[indices] * m[:, None]).sum(axis=0) / node.mass
    if len(indices) <= max_per_leaf:
        node.indices = list(indices)
        return node
    node.is_leaf = False
    D = pos.shape[1]
    half = size / 2.0
    offsets = np.array(
        np.meshgrid(*[[-half / 2, half / 2]] * D, indexing='ij')
    ).reshape(D, -1).T
    buckets = {tuple(o): [] for o in offsets}
    rel = pos[indices] - center
    for k, idx in enumerate(indices):
        key = tuple(np.where(rel[k] >= 0, half / 2, -half / 2))
        buckets[key].append(int(idx))
    for off in offsets:
        child_center = center + off
        child = build_tree(pos, mass, child_center, half,
                           np.array(buckets[tuple(off)], dtype=int),
                           max_per_leaf)
        node.children.append(child)
    return node


def accel_from_tree(node, pos_i, G=1.0, theta=0.5, eps=0.05):
    if node.mass == 0.0:
        return np.zeros_like(pos_i)
    d = node.com - pos_i
    r2 = float(np.dot(d, d)) + eps ** 2
    r = np.sqrt(r2)
    if node.is_leaf or (node.size / r) < theta:
        return G * node.mass * d / (r2 * r)
    a = np.zeros_like(pos_i)
    for child in node.children:
        a += accel_from_tree(child, pos_i, G, theta, eps)
    return a


def accelerations_bh(pos, mass, G=1.0, theta=0.5, eps=0.05,
                     root_size=None):
    N, D = pos.shape
    if root_size is None:
        root_size = 2.0 * float(np.max(np.ptp(pos, axis=0))) + 1e-6
    center = pos.mean(axis=0)
    tree = build_tree(pos, mass, center, root_size,
                      np.arange(N, dtype=int), max_per_leaf=1)
    acc = np.empty_like(pos)
    for i in range(N):
        acc[i] = accel_from_tree(tree, pos[i], G, theta, eps)
    return acc


def wrap(pos, L):
    return pos - L * np.floor(pos / L)


def leapfrog_step(pos, vel, mass, dt, G=1.0, eps=0.05,
                  use_tree=True, theta=0.5, L=None):
    if use_tree:
        acc = accelerations_bh(pos, mass, G, theta, eps)
    else:
        acc = accelerations_direct(pos, mass, G, eps, L)
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    if L is not None:
        pos_new = wrap(pos_new, L)
    if use_tree:
        acc_new = accelerations_bh(pos_new, mass, G, theta, eps)
    else:
        acc_new = accelerations_direct(pos_new, mass, G, eps, L)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new


def verlet_step(pos, pos_prev, mass, dt, G=1.0, eps=0.05,
                use_tree=True, theta=0.5, L=None):
    if use_tree:
        acc = accelerations_bh(pos, mass, G, theta, eps)
    else:
        acc = accelerations_direct(pos, mass, G, eps, L)
    pos_new = 2.0 * pos - pos_prev + dt ** 2 * acc
    if L is not None:
        pos_new = wrap(pos_new, L)
    return pos_new


def total_energy(pos, vel, mass, G=1.0, eps=0.05, L=None):
    K = 0.5 * float(np.sum(mass * np.sum(vel ** 2, axis=1)))
    diff = pos[None, :, :] - pos[:, None, :]
    if L is not None:
        diff = diff - L * np.round(diff / L)
    r2 = np.sum(diff ** 2, axis=-1) + eps ** 2
    np.fill_diagonal(r2, np.inf)
    U = -0.5 * G * float(np.sum(
        (mass[:, None] * mass[None, :]) / np.sqrt(r2)))
    return K + U, K, U


def init_disk(N, D=2, seed=42, G=1.0):
    rng = np.random.default_rng(seed)
    mass = rng.uniform(0.5, 1.5, size=N)
    if D == 2:
        R = rng.uniform(0.0, 1.0, size=N) ** 0.7
        phi = rng.uniform(0, 2 * np.pi, size=N)
        pos = np.stack([R * np.cos(phi), R * np.sin(phi)], axis=1)
    else:
        R = rng.uniform(0.0, 1.0, size=N) ** 0.7
        u = rng.uniform(-1, 1, size=N)
        phi = rng.uniform(0, 2 * np.pi, size=N)
        s = np.sqrt(1 - u ** 2)
        pos = np.stack([R * s * np.cos(phi),
                        R * s * np.sin(phi),
                        R * u], axis=1)
    M_tot = mass.sum()
    com = (pos * mass[:, None]).sum(axis=0) / M_tot
    pos -= com
    r = np.linalg.norm(pos, axis=1) + 1e-6
    v_circ = np.sqrt(G * M_tot / r) * 0.8
    if D == 2:
        vel = np.stack([-pos[:, 1], pos[:, 0]], axis=1) / r[:, None] \
              * v_circ[:, None]
    else:
        r_xy = np.linalg.norm(pos[:, :2], axis=1) + 1e-6
        tang = np.stack([-pos[:, 1] / r_xy,
                         pos[:, 0] / r_xy,
                         np.zeros_like(r_xy)], axis=1)
        vel = tang * v_circ[:, None]
    vel += rng.normal(0, 0.05, size=vel.shape)
    return pos, vel, mass


def save_snapshot(pos, mass, filename, title, L_lim=None, dpi=150):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    if L_lim is None:
        L_lim = 1.2 * np.max(np.abs(pos))
    ax.set_xlim(-L_lim, L_lim)
    ax.set_ylim(-L_lim, L_lim)
    sizes = 5.0 * (mass / mass.max())
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c='white', alpha=0.9)
    ax.set_title(title, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Сохранён снимок: {filename}")


def run_dual_simulation(N=300, D=2, n_steps=600, dt=0.005,
                        G=1.0, eps=0.05, theta=0.5,
                        seed=42, L=None,
                        save_initial=None, save_final=None):
    """
    Запускает две симуляции параллельно:
      - PP  (прямое суммирование) — эталон
      - BH  (Barnes-Hut)           — приближение
    Возвращает траектории и энергии обеих систем.
    """
    print(f"Двойная симуляция: N={N}, D={D}, steps={n_steps}, dt={dt}, "
          f"theta={theta}")

    pos_pp, vel_pp, mass = init_disk(N, D, seed, G)
    pos_bh = pos_pp.copy()
    vel_bh = vel_pp.copy()

    if save_initial is not None and D == 2:
        save_snapshot(pos_pp, mass, save_initial,
                      "Шаг 0 (начальное состояние)")

    traj_pp = np.empty((n_steps + 1, N, D))
    traj_bh = np.empty((n_steps + 1, N, D))
    traj_pp[0] = pos_pp
    traj_bh[0] = pos_bh

    E_pp = np.empty(n_steps + 1)
    E_bh = np.empty(n_steps + 1)
    E_pp[0], _, _ = total_energy(pos_pp, vel_pp, mass, G, eps, L)
    E_bh[0], _, _ = total_energy(pos_bh, vel_bh, mass, G, eps, L)

    acc_pp_0 = accelerations_direct(pos_pp, mass, G, eps, L)
    acc_bh_0 = accelerations_bh(pos_bh, mass, G, theta, eps)

    pos_prev_pp = pos_pp - dt * vel_pp + 0.5 * dt ** 2 * acc_pp_0
    pos_prev_bh = pos_bh - dt * vel_bh + 0.5 * dt ** 2 * acc_bh_0
    if L is not None:
        pos_prev_pp = wrap(pos_prev_pp, L)
        pos_prev_bh = wrap(pos_prev_bh, L)

    for step in range(n_steps):
        pos_new_pp = verlet_step(pos_pp, pos_prev_pp, mass, dt,
                                  G, eps, use_tree=False, theta=theta, L=L)
        vel_pp = (pos_new_pp - pos_prev_pp) / (2 * dt)
        pos_prev_pp, pos_pp = pos_pp, pos_new_pp
        traj_pp[step + 1] = pos_pp
        E_pp[step + 1], _, _ = total_energy(pos_pp, vel_pp, mass, G, eps, L)

        pos_new_bh = verlet_step(pos_bh, pos_prev_bh, mass, dt,
                                  G, eps, use_tree=True, theta=theta, L=L)
        vel_bh = (pos_new_bh - pos_prev_bh) / (2 * dt)
        pos_prev_bh, pos_bh = pos_bh, pos_new_bh
        traj_bh[step + 1] = pos_bh
        E_bh[step + 1], _, _ = total_energy(pos_bh, vel_bh, mass, G, eps, L)

    if save_final is not None and D == 2:
        base = save_final.rsplit(".", 1)[0]
        save_snapshot(pos_pp, mass, f"{base}_pp.png",
                      f"Шаг {n_steps} (эталон PP)")
        save_snapshot(pos_bh, mass, f"{base}_bh.png",
                      f"Шаг {n_steps} (приближение BH)")

    return traj_pp, traj_bh, E_pp, E_bh


def animate_dual(traj_pp, traj_bh, interval=30, L_lim=None,
                 save_gif=None, gif_fps=30):
    """
    Анимация: белые точки — эталон (PP), красные — приближение (BH).
    """
    n_frames, N, D = traj_pp.shape
    if D != 2:
        raise ValueError("Анимация работает только для D=2")

    if L_lim is None:
        all_pos = np.concatenate([traj_pp, traj_bh], axis=0)
        L_lim = 1.2 * np.max(np.abs(all_pos))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.set_xlim(-L_lim, L_lim)
    ax.set_ylim(-L_lim, L_lim)
    ax.set_facecolor('black')

    scat_pp = ax.scatter(traj_pp[0, :, 0], traj_pp[0, :, 1],
                          s=10, c='white', alpha=0.7,
                          edgecolors='none', label='PP (эталон)')
    scat_bh = ax.scatter(traj_bh[0, :, 0], traj_bh[0, :, 1],
                          s=10, c='red', alpha=0.7,
                          edgecolors='none', label='BH (приближение)')

    ax.legend(loc='upper right', facecolor='black',
              edgecolor='white', labelcolor='white', framealpha=0.8)

    def update(frame):
        scat_pp.set_offsets(traj_pp[frame])
        scat_bh.set_offsets(traj_bh[frame])
        ax.set_title(f"Шаг {frame}/{n_frames - 1}", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        return scat_pp, scat_bh

    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=interval, blit=True)

    if save_gif is not None:
        os.makedirs(os.path.dirname(save_gif) or ".", exist_ok=True)
        ani.save(save_gif, writer='pillow', fps=gif_fps)
        print(f"Сохранена анимация: {save_gif}")

    plt.show()
    return ani


def plot_divergence(traj_pp, traj_bh, save_path=None):
    """График |r_PP - r_BH| по времени: среднее и максимум."""
    diff = np.linalg.norm(traj_pp - traj_bh, axis=-1)  # (n_frames, N)
    mean_div = diff.mean(axis=1)
    max_div = diff.max(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mean_div, label='Среднее отклонение', color='steelblue')
    ax.plot(max_div, label='Максимальное отклонение', color='crimson')
    ax.set_xlabel('Шаг')
    ax.set_ylabel('|Δr|')
    ax.set_title('Расхождение PP и Barnes–Hut во времени')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранён график расхождения: {save_path}")

    plt.show()


def plot_energy_dual(E_pp, E_bh, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(E_pp / E_pp[0], label='PP (эталон)', color='steelblue')
    ax.plot(E_bh / E_bh[0], label='BH (приближение)', color='crimson',
            linestyle='--')
    ax.set_xlabel('Шаг')
    ax.set_ylabel('E(t) / E(0)')
    ax.set_title('Сохранение энергии: PP и Barnes–Hut')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранён график энергии: {save_path}")

    plt.show()


def compare_pp_bh(N=300, D=3, theta=0.5, eps=0.05, seed=1):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1, size=(N, D))
    mass = rng.uniform(0.5, 1.5, size=N)
    acc_pp = accelerations_direct(pos, mass, G=1.0, eps=eps)
    acc_bh = accelerations_bh(pos, mass, G=1.0, theta=theta, eps=eps)
    num = np.linalg.norm(acc_pp - acc_bh, axis=1)
    den = np.linalg.norm(acc_pp, axis=1) + 1e-12
    err = num / den
    print(f"\n[Сравнение PP vs Barnes-Hut, N={N}, theta={theta}]")
    print(f"  Средняя относительная ошибка : {err.mean()*100:.4f}%")
    print(f"  Максимальная ошибка          : {err.max()*100:.4f}%")
    return err


if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)

    compare_pp_bh(N=300, D=3, theta=0.5)

    N     = 300
    D     = 2
    steps = 600
    dt    = 0.005

    traj_pp, traj_bh, E_pp, E_bh = run_dual_simulation(
        N=N, D=D, n_steps=steps, dt=dt,
        G=1.0, eps=0.05, theta=0.5,
        seed=42, L=None,
        save_initial="docs/nbody_initial.png",
        save_final="docs/nbody_final.png"
    )

    plot_energy_dual(E_pp, E_bh,
                     save_path="docs/nbody_energy.png")
    plot_divergence(traj_pp, traj_bh,
                    save_path="docs/nbody_divergence.png")
    animate_dual(traj_pp, traj_bh, interval=30,
                 save_gif="docs/nbody_animation.gif")