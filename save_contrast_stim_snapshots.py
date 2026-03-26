import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")


def generate_grating_frame(
    rows: int,
    cols: int,
    theta_deg: float,
    cpd: float,
    contrast: float,
    degrees_per_pixel: float = 1.0,
    phase_deg: float = 0.0,
    mean_luminance: float = 0.5,
):
    """Generate a static sinusoidal grating image.

    The output is in [0, 1] range representing luminance.
    """
    # Coordinates in degrees of visual angle
    y = (np.arange(rows) - rows / 2) * degrees_per_pixel
    x = (np.arange(cols) - cols / 2) * degrees_per_pixel
    X, Y = np.meshgrid(x, y)

    # Orientation
    theta = np.deg2rad(theta_deg)
    # Spatial frequency in cycles per degree
    fx = cpd * np.cos(theta)
    fy = cpd * np.sin(theta)

    phase = np.deg2rad(phase_deg)
    grating = np.sin(2 * np.pi * (fx * X + fy * Y) + phase)

    # Michelson contrast scaling around mean luminance
    img = mean_luminance + (contrast / 2.0) * grating
    img = np.clip(img, 0.0, 1.0)
    return img


def format_theta_for_fname(theta_deg: float) -> str:
    """
    Format an orientation angle for filenames.

    Examples:
      0.0 -> "0"
      45.0 -> "45"
      22.5 -> "22p5"
      -22.5 -> "m22p5"
    """
    # Avoid float representation noise in filenames.
    theta = float(np.round(theta_deg, 6))

    if np.isfinite(theta) and np.isclose(theta, round(theta), atol=1e-6):
        return str(int(round(theta)))

    sign = "m" if theta < 0 else ""
    theta_abs = abs(theta)
    s = f"{theta_abs:.6f}".rstrip("0").rstrip(".")
    return sign + s.replace(".", "p")


def main():
    parser = argparse.ArgumentParser(
        description="Save snapshots of contrast drifting grating stimuli."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures/stimulus_snapshots",
        help="Output directory for saved images.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.0,
        help="Orientation angle in degrees.",
    )
    parser.add_argument(
        "--thetas",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional list of orientation angles in degrees. "
            "If provided, overrides --theta."
        ),
    )
    parser.add_argument(
        "--use_dg_angles",
        action="store_true",
        help="Use the canonical 8 drifting-gratings angles (0..315 step 45).",
    )
    parser.add_argument(
        "--contrasts",
        type=float,
        nargs="*",
        default=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80],
        help="List of contrast values (0-1).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=80,
        help="Image rows, should match config row_size.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=120,
        help="Image cols, should match config col_size.",
    )
    parser.add_argument(
        "--cpd",
        type=float,
        default=0.04,
        help="Cycles per degree (spatial frequency).",
    )
    parser.add_argument(
        "--degrees_per_pixel",
        type=float,
        default=1.0,
        help="Degrees per pixel to match model config.",
    )
    parser.add_argument(
        "--phase",
        type=float,
        default=0.0,
        help="Phase in degrees for the snapshot.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.thetas is not None and len(args.thetas) > 0:
        thetas = list(args.thetas)
    elif args.use_dg_angles:
        # Keep this import local so the script stays usable as a standalone utility.
        from stimulus_trials import DriftingGratingsStimulus

        thetas = [float(a) for a in DriftingGratingsStimulus().angles]
    else:
        thetas = [float(args.theta)]

    n_saved = 0
    for theta in thetas:
        theta_token = format_theta_for_fname(theta)
        for c in args.contrasts:
            img = generate_grating_frame(
                rows=args.rows,
                cols=args.cols,
                theta_deg=theta,
                cpd=args.cpd,
                contrast=c,
                degrees_per_pixel=args.degrees_per_pixel,
                phase_deg=args.phase,
            )
            fig, ax = plt.subplots(figsize=(4, 2.7), dpi=200)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper")
            ax.axis("off")
            fig.tight_layout(pad=0)
            fpath = outdir / f"grating_theta{theta_token}_contrast{c:.2f}.png"
            fig.savefig(fpath, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            n_saved += 1

    print(f"Saved {n_saved} snapshots to {outdir}")


if __name__ == "__main__":
    main()
