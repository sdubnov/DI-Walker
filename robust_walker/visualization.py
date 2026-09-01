"""Animation helpers for quadruped rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import BODY


def animate_rollout(rollout: dict[str, np.ndarray], outfile: str | Path, title: str = "Quadruped coordination") -> None:
    """Render a single rollout with Matplotlib."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    state = rollout["state"]
    target = rollout["target"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(target[:, 0], target[:, 1], "--", label="Target")
    pad = 0.8
    ax.set_xlim(min(state[:, 0].min(), target[:, 0].min()) - pad, max(state[:, 0].max(), target[:, 0].max()) + pad)
    ax.set_ylim(min(state[:, 1].min(), target[:, 1].min()) - pad, max(state[:, 1].max(), target[:, 1].max()) + pad)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("World x")
    ax.set_ylabel("World y")
    trail, = ax.plot([], [], label="Path")
    body, = ax.plot([], [], "o-")
    legs = [ax.plot([], [], lw=2)[0] for _ in range(4)]
    ax.legend()

    def update(k: int):
        ph = state[k, 2]
        center = state[k, :2]
        c, sn = np.cos(ph), np.sin(ph)
        points = np.array([center + [c * r[0] - sn * r[1], sn * r[0] + c * r[1]] for r in BODY])
        poly = points[[0, 1, 3, 2, 0]]
        body.set_data(poly[:, 0], poly[:, 1])
        trail.set_data(state[: k + 1, 0], state[: k + 1, 1])
        forward = np.array([c, sn])
        for i, limb in enumerate(legs):
            end = points[i] + 0.28 * rollout["act"][k, i] * forward
            limb.set_data([points[i, 0], end[0]], [points[i, 1], end[1]])
        return [trail, body, *legs]

    FuncAnimation(fig, update, frames=range(0, len(state), 2), interval=70).save(
        outfile,
        writer=PillowWriter(fps=14),
    )
    plt.close(fig)


def animate_comparison(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    outfile: str | Path,
    left_title: str = "Capacity",
    right_title: str = "Sensor-Comm",
    title: str = "Failure recovery comparison",
) -> None:
    """Render a side-by-side Matplotlib comparison of two rollouts."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    rollouts = [left, right]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    fig.suptitle(title)

    all_state = np.vstack([r["state"] for r in rollouts])
    all_target = np.vstack([r["target"] for r in rollouts])
    pad = 0.8
    xlim = (
        min(all_state[:, 0].min(), all_target[:, 0].min()) - pad,
        max(all_state[:, 0].max(), all_target[:, 0].max()) + pad,
    )
    ylim = (
        min(all_state[:, 1].min(), all_target[:, 1].min()) - pad,
        max(all_state[:, 1].max(), all_target[:, 1].max()) + pad,
    )

    artists = []
    for ax, rollout_data, panel_title in zip(axes, rollouts, [left_title, right_title]):
        ax.plot(rollout_data["target"][:, 0], rollout_data["target"][:, 1], "--", color="0.45", label="Target")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(panel_title)
        ax.set_xlabel("World x")
        ax.grid(alpha=0.18)
        trail, = ax.plot([], [], color="#1f77b4", lw=2, label="Path")
        body, = ax.plot([], [], "o-", color="#111111", lw=2)
        legs = [ax.plot([], [], lw=2)[0] for _ in range(4)]
        artists.append((trail, body, legs))
    axes[0].set_ylabel("World y")
    axes[0].legend(loc="upper left")

    def update(k: int):
        frame_artists = []
        for rollout_data, (trail, body, legs) in zip(rollouts, artists):
            state = rollout_data["state"]
            ph = state[k, 2]
            center = state[k, :2]
            c, sn = np.cos(ph), np.sin(ph)
            points = np.array([center + [c * r[0] - sn * r[1], sn * r[0] + c * r[1]] for r in BODY])
            poly = points[[0, 1, 3, 2, 0]]
            body.set_data(poly[:, 0], poly[:, 1])
            trail.set_data(state[: k + 1, 0], state[: k + 1, 1])
            forward = np.array([c, sn])
            for i, limb in enumerate(legs):
                end = points[i] + 0.28 * rollout_data["act"][k, i] * forward
                limb.set_data([points[i, 0], end[0]], [points[i, 1], end[1]])
            frame_artists.extend([trail, body, *legs])
        return frame_artists

    n_frames = min(len(left["state"]), len(right["state"]))
    FuncAnimation(fig, update, frames=range(0, n_frames, 2), interval=70).save(
        outfile,
        writer=PillowWriter(fps=14),
    )
    plt.close(fig)


def animate_comparison_pillow(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    outfile: str | Path,
    title: str = "Failure recovery comparison",
    left_title: str = "Capacity",
    right_title: str = "Sensor-Comm",
    width: int = 1200,
    height: int = 540,
    step: int = 2,
) -> None:
    """Render a side-by-side GIF without Matplotlib font-cache dependencies."""

    from PIL import Image, ImageDraw, ImageFont

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    rollouts = [left, right]
    all_state = np.vstack([r["state"] for r in rollouts])
    all_target = np.vstack([r["target"] for r in rollouts])
    pad = 0.8
    xmin = min(all_state[:, 0].min(), all_target[:, 0].min()) - pad
    xmax = max(all_state[:, 0].max(), all_target[:, 0].max()) + pad
    ymin = min(all_state[:, 1].min(), all_target[:, 1].min()) - pad
    ymax = max(all_state[:, 1].max(), all_target[:, 1].max()) + pad
    span = max(xmax - xmin, ymax - ymin)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    xmin, xmax = cx - 0.5 * span, cx + 0.5 * span
    ymin, ymax = cy - 0.5 * span, cy + 0.5 * span

    font = ImageFont.load_default()
    panel_w = width // 2
    margin = 48
    top = 82
    bottom = 32
    panel_h = height - top - bottom

    def project(point, panel_index):
        x, y = point
        px = panel_index * panel_w + margin + (x - xmin) / (xmax - xmin) * (panel_w - 2 * margin)
        py = top + panel_h - (y - ymin) / (ymax - ymin) * panel_h
        return int(px), int(py)

    def draw_polyline(draw, points, panel_index, fill, width_px=2, dashed=False):
        coords = [project(p, panel_index) for p in points]
        if len(coords) < 2:
            return
        if not dashed:
            draw.line(coords, fill=fill, width=width_px)
            return
        for i in range(0, len(coords) - 1, 4):
            draw.line([coords[i], coords[i + 1]], fill=fill, width=width_px)

    frames = []
    n_frames = min(len(left["state"]), len(right["state"]))
    for k in range(0, n_frames, step):
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((margin, 18), title, fill=(20, 20, 20), font=font)
        draw.text((margin, 52), left_title, fill=(20, 20, 20), font=font)
        draw.text((panel_w + margin, 52), right_title, fill=(20, 20, 20), font=font)
        draw.line([(panel_w, 0), (panel_w, height)], fill=(220, 220, 220), width=1)

        for panel_index, rollout_data in enumerate(rollouts):
            target = rollout_data["target"]
            state = rollout_data["state"]
            draw_polyline(draw, target[:, :2], panel_index, fill=(135, 135, 135), width_px=2, dashed=True)
            draw_polyline(draw, state[: k + 1, :2], panel_index, fill=(35, 105, 170), width_px=3)

            ph = state[k, 2]
            center = state[k, :2]
            c, sn = np.cos(ph), np.sin(ph)
            points = np.array([center + [c * r[0] - sn * r[1], sn * r[0] + c * r[1]] for r in BODY])
            body = points[[0, 1, 3, 2, 0]]
            draw.line([project(p, panel_index) for p in body], fill=(20, 20, 20), width=3)
            forward = np.array([c, sn])
            for i, point in enumerate(points):
                end = point + 0.28 * rollout_data["act"][k, i] * forward
                draw.line([project(point, panel_index), project(end, panel_index)], fill=(205, 80, 40), width=3)
                x, y = project(point, panel_index)
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(20, 20, 20))

        frames.append(img)

    frames[0].save(
        outfile,
        save_all=True,
        append_images=frames[1:],
        duration=70,
        loop=0,
        optimize=True,
    )


def animate_spacious_summary_pillow(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    outfile: str | Path,
    title: str,
    left_title: str,
    right_title: str,
    left_error: float,
    right_error: float,
    limb_fault_step: int = 65,
    cc_period: int = 30,
    cc_on_steps: int = 10,
    width: int = 1440,
    height: int = 1080,
    step: int = 2,
    duration: int = 110,
) -> None:
    """Render the large, slideshow-readable intermittent-CC/L4-slip GIF."""

    from PIL import Image, ImageDraw, ImageFont

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    rollouts = [left, right]
    all_state = np.vstack([r["state"] for r in rollouts])
    all_target = np.vstack([r["target"] for r in rollouts])
    pad = 0.9
    xmin = min(all_state[:, 0].min(), all_target[:, 0].min()) - pad
    xmax = max(all_state[:, 0].max(), all_target[:, 0].max()) + pad
    ymin = min(all_state[:, 1].min(), all_target[:, 1].min()) - pad
    ymax = max(all_state[:, 1].max(), all_target[:, 1].max()) + pad
    span = max(xmax - xmin, ymax - ymin)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    xmin, xmax = cx - 0.5 * span, cx + 0.5 * span
    ymin, ymax = cy - 0.5 * span, cy + 0.5 * span

    def font(size: int):
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_font = font(46)
    panel_font = font(34)
    label_font = font(28)
    small_font = font(24)
    status_font = font(38)
    panel_w = width // 2
    plot_left_margin = 86
    plot_right_margin = 86
    top = 305
    bottom = 190
    panel_h = height - top - bottom

    def project(point, panel_index):
        x, y = point
        px = panel_index * panel_w + plot_left_margin + (x - xmin) / (xmax - xmin) * (
            panel_w - plot_left_margin - plot_right_margin
        )
        py = top + panel_h - (y - ymin) / (ymax - ymin) * panel_h
        return int(px), int(py)

    def draw_polyline(draw, points, panel_index, fill, width_px=3, dashed=False):
        coords = [project(p, panel_index) for p in points]
        if len(coords) < 2:
            return
        if not dashed:
            draw.line(coords, fill=fill, width=width_px)
            return
        for i in range(0, len(coords) - 1, 5):
            draw.line([coords[i], coords[i + 1]], fill=fill, width=width_px)

    def draw_status(draw, x, y, lines):
        box_w = width - 2 * x
        box_h = 82
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=14, fill=(248, 248, 248), outline=(170, 170, 170), width=3)
        col_w = box_w // len(lines)
        for i, line in enumerate(lines):
            draw.text((x + 28 + col_w * i, y + 20), line, fill=(20, 20, 20), font=status_font)

    frames = []
    n_frames = min(len(left["state"]), len(right["state"]))
    for k in range(0, n_frames, step):
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((70, 32), title, fill=(18, 18, 18), font=title_font)
        draw.text((70, 104), "Large-layout replay: intermittent central control + intermittent L4 slip", fill=(60, 60, 60), font=label_font)
        draw.line((panel_w, 246, panel_w, height - 42), fill=(220, 220, 220), width=2)

        panel_titles = [left_title, right_title]
        errors = [left_error, right_error]
        for panel_index, (rollout_data, panel_title, error) in enumerate(zip(rollouts, panel_titles, errors)):
            x0 = panel_index * panel_w + 70
            draw.text((x0, 250), panel_title, fill=(18, 18, 18), font=panel_font)
            draw.text((x0, height - 146), f"Late path error: {error:.3f}", fill=(18, 18, 18), font=label_font)
            draw.line((x0, height - 102, x0 + 52, height - 102), fill=(135, 135, 135), width=4)
            draw.text((x0 + 66, height - 116), "Target", fill=(70, 70, 70), font=small_font)
            draw.line((x0, height - 64, x0 + 52, height - 64), fill=(35, 105, 170), width=5)
            draw.text((x0 + 66, height - 78), "Body path", fill=(70, 70, 70), font=small_font)

            target = rollout_data["target"]
            state = rollout_data["state"]
            draw_polyline(draw, target[:, :2], panel_index, fill=(135, 135, 135), width_px=4, dashed=True)
            draw_polyline(draw, state[: k + 1, :2], panel_index, fill=(35, 105, 170), width_px=5)

            ph = state[k, 2]
            center = state[k, :2]
            c, sn = np.cos(ph), np.sin(ph)
            points = np.array([center + [c * r[0] - sn * r[1], sn * r[0] + c * r[1]] for r in BODY])
            body = points[[0, 1, 3, 2, 0]]
            draw.line([project(p, panel_index) for p in body], fill=(20, 20, 20), width=7)
            forward = np.array([c, sn])
            for i, point in enumerate(points):
                end = point + 0.38 * rollout_data["act"][k, i] * forward
                color = (205, 80, 40) if i != 3 else (170, 60, 160)
                draw.line([project(point, panel_index), project(end, panel_index)], fill=color, width=7)
                x, y = project(point, panel_index)
                radius = 9 if i != 3 else 11
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(20, 20, 20))

        cc_on = (k % cc_period) < cc_on_steps
        slip_on = k >= limb_fault_step and ((k - limb_fault_step) % 24) < 10
        status_lines = [
            f"step: {k:03d}",
            f"CC: {'on' if cc_on else 'off'}",
            f"L4 slip: {'active' if slip_on else 'recovered'}",
        ]
        draw_status(draw, 70, 152, status_lines)
        frames.append(img)

    frames[0].save(
        outfile,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
