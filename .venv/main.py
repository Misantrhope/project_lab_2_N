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
    fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Сохранён снимок: {filename}")


def run_simulation(N=500, D=2, n_steps=800, dt=0.005,
                   G=1.0, eps=0.05, theta=0.5,
                   use_tree=False, integrator="verlet",
                   seed=42, L=None,
                   save_initial=None, save_final=None):
    print(f"Запуск: N={N}, D={D}, steps={n_steps}, dt={dt}, "
          f"tree={use_tree}, integrator={integrator}")
    pos, vel, mass = init_disk(N, D, seed, G)

    if save_initial is not None and D == 2:
        save_snapshot(pos, mass, save_initial,
                      title=f"Шаг 0 / {n_steps}")

    trajectory = np.empty((n_steps + 1, N, D))
    trajectory[0] = pos
    E0, _, _ = total_energy(pos, vel, mass, G, eps, L)
    energies = np.empty(n_steps + 1)
    energies[0] = E0

    if integrator == "verlet":
        if use_tree:
            acc0 = accelerations_bh(pos, mass, G, theta, eps)
        else:
            acc0 = accelerations_direct(pos, mass, G, eps, L)
        pos_prev = pos - dt * vel + 0.5 * dt ** 2 * acc0
        if L is not None:
            pos_prev = wrap(pos_prev, L)

    for step in range(n_steps):
        if integrator == "verlet":
            pos_new = verlet_step(pos, pos_prev, mass, dt,
                                  G, eps, use_tree, theta, L)
            vel = (pos_new - pos_prev) / (2 * dt)
            pos_prev, pos = pos, pos_new
        else:
            pos, vel = leapfrog_step(pos, vel, mass, dt,
                                     G, eps, use_tree, theta, L)
        trajectory[step + 1] = pos
        E, _, _ = total_energy(pos, vel, mass, G, eps, L)
        energies[step + 1] = E

    if save_final is not None and D == 2:
        save_snapshot(pos, mass, save_final,
                      title=f"Шаг {n_steps} / {n_steps}")

    print(f"E0 = {energies[0]:.6f}, E_end = {energies[-1]:.6f}, "
          f"ΔE/E0 = {abs((energies[-1]-energies[0])/energies[0])*100:.4f}%")
    return trajectory, energies


def animate(trajectory, interval=30, L_lim=None):
    n_frames, N, D = trajectory.shape
    if D != 2:
        raise ValueError("Анимация работает только для D=2")
    if L_lim is None:
        L_lim = 1.2 * np.max(np.abs(trajectory))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-L_lim, L_lim)
    ax.set_ylim(-L_lim, L_lim)
    ax.set_facecolor('black')
    scat = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1],
                      s=5, c='white', alpha=0.9)

    def update(frame):
        scat.set_offsets(trajectory[frame])
        ax.set_title(f"Шаг {frame}/{n_frames-1}", color='white')
        ax.tick_params(colors='white')
        return scat,

    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=interval, blit=True)
    plt.show()
    return ani


def plot_energy(energies):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(energies / energies[0], color='steelblue')
    ax.set_xlabel('Шаг')
    ax.set_ylabel('E(t) / E(0)')
    ax.set_title('Сохранение полной энергии')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_pp_bh(N=500, D=3, theta=0.5, eps=0.05, seed=1):
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
    compare_pp_bh(N=500, D=3, theta=0.5)

    N        = 500
    D        = 2
    steps    = 600
    dt       = 0.005
    use_tree = False

    traj, E = run_simulation(
        N=N, D=D, n_steps=steps, dt=dt,
        G=1.0, eps=0.05, theta=0.5,
        use_tree=use_tree, integrator="verlet",
        seed=42, L=None,
        save_initial="nbody_initial.png",
        save_final="nbody_final.png"
    )

    plot_energy(E)
    animate(traj, interval=30)