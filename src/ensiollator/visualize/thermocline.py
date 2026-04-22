import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from pde import FileStorage, FieldCollection


def plot_thermocline_timeseries(
    data_dir: str,
    out_path: str | None = None,
):
    """Plot spatially averaged thermocline height vs time.

    Draws three lines on the same axes:
        - h_mean : domain-wide spatial average
        - h_W    : western-half average  (x < basin midpoint)
        - h_E    : eastern-half average  (x > basin midpoint)

    Args:
        data_dir: Directory containing ``simulation.hdf5``.
        out_path: Output path for the saved figure. Defaults to
            ``<data_dir>/thermocline_timeseries.png``.
    """
    if out_path is None:
        out_path = os.path.join(data_dir, "thermocline_timeseries.png")

    storage = FileStorage(os.path.join(data_dir, "simulation.hdf5"), write_mode="readonly")
    times = np.array(list(storage.times))

    h_mean = np.empty(len(times))
    h_W    = np.empty(len(times))
    h_E    = np.empty(len(times))

    for i in range(len(times)):
        h = FieldCollection(storage[i])[2].data  # type: ignore[arg-type]
        Nx = h.shape[0]
        h_mean[i] = h.mean()
        h_W[i]    = h[: int(Nx * 0.5), :].mean()
        h_E[i]    = h[int(Nx * 0.5) :, :].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, h_mean, label=r"$\bar{h}$  (domain mean)")
    ax.plot(times, h_W,    label=r"$h_W$  (west)")
    ax.plot(times, h_E,    label=r"$h_E$  (east)")
    ax.set_xlabel("t (non-dim)")
    ax.set_ylabel("h (non-dim)")
    ax.set_title("Spatially averaged thermocline depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_thermocline_video(
    data_dir: str,
    out_path: str | None = None,
    steps: list[int] | None = None,
    coarsen_factor: int = 1,
    fps: int = 10,
    dpi: int = 150,
):
    """Animate thermocline depth from a saved simulation.

    Args:
        data_dir: Directory containing ``simulation.hdf5``.
        out_path: Output ``.mp4`` path. Defaults to ``<data_dir>/thermocline.mp4``.
        steps: Frame indices to include. Defaults to all frames.
        coarsen_factor: Keep every Nth frame from ``steps`` (or all frames if
            ``steps`` is None). Defaults to 1 (no coarsening).
        fps: Frames per second of the output video.
        dpi: Resolution of the output video.
    """
    if out_path is None:
        out_path = os.path.join(data_dir, "thermocline.mp4")

    storage = FileStorage(os.path.join(data_dir, "simulation.hdf5"), write_mode="readonly")
    all_times = list(storage.times)

    base_indices = list(range(len(all_times))) if steps is None else steps
    indices = base_indices[::coarsen_factor]

    times = [all_times[i] for i in indices]
    h_frames = [
        FieldCollection(storage[i])[2].data  # type: ignore[arg-type]
        for i in tqdm(indices, desc="Loading frames")
    ]

    first: FieldCollection = storage[0]  # type: ignore[assignment]
    coords = first.grid.cell_coords
    X = coords[:, :, 0]
    Y = coords[:, :, 1]

    h_all = np.stack(h_frames)
    zlim = (h_all.min(), h_all.max())
    z_pad = 0.05 * (zlim[1] - zlim[0]) or 0.1
    zlim = (zlim[0] - z_pad, zlim[1] + z_pad)

    x_range = float(X.max() - X.min())
    y_range = float(Y.max() - Y.min())

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([x_range, y_range, y_range * 0.4], zoom=1.4)  # type: ignore[arg-type]
    ax.view_init(elev=10, azim=35)

    surf = ax.plot_surface(X, Y, h_frames[0], cmap='RdBu_r',
                           vmin=zlim[0], vmax=zlim[1], linewidth=0, antialiased=False)

    title = ax.set_title(f"Thermocline depth  t = {times[0]:.2f}")
    ax.set_xlabel("x (non-dim)", labelpad=20)
    ax.set_ylabel("y (non-dim)", labelpad=20)
    ax.set_zlabel("h (non-dim)", labelpad=1)
    ax.set_zlim(*zlim)
    fig.tight_layout()

    pbar = tqdm(total=len(times), desc="Animating Thermocline Depth")

    def update(i):
        pbar.update(1)
        nonlocal surf
        surf.remove()
        surf = ax.plot_surface(X, Y, h_frames[i], cmap='RdBu_r',
                               vmin=zlim[0], vmax=zlim[1], linewidth=0, antialiased=False)
        title.set_text(f"Thermocline depth  t = {times[i]:.2f}")
        return surf,

    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(out_path, writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
    pbar.close()
