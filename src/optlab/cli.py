from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from .metrics import mse, psnr
from .problems import add_noise, make_phantom
from .solvers import denoise_with_accelerated_gradient, denoise_with_admm, denoise_with_gradient_descent


def _save_image(path: Path, image: np.ndarray, title: str) -> None:
    plt.figure(figsize=(4.4, 4.4))
    plt.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_benchmark(path: Path, ground_truth: np.ndarray, noisy: np.ndarray, results: list[tuple[str, np.ndarray]]) -> None:
    columns = 2 + len(results)
    fig, axes = plt.subplots(1, columns, figsize=(3.6 * columns, 3.6))

    axes[0].imshow(ground_truth, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Ground truth")
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Noisy input")
    axes[1].axis("off")

    for axis, (name, image) in zip(axes[2:], results):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(name.replace("_", " ").title())
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_results_table(path: Path, ground_truth: np.ndarray, outputs: list[tuple[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "mse", "psnr", "iterations", "objective", "residual", "runtime_seconds", "converged"])
        for name, result in outputs:
            writer.writerow(
                [
                    name,
                    f"{mse(ground_truth, result.image):.8f}",
                    f"{psnr(ground_truth, result.image):.4f}",
                    result.iterations,
                    f"{result.objective_history[-1]:.8f}" if result.objective_history else "",
                    f"{result.residual_history[-1]:.8e}" if result.residual_history else "",
                    f"{result.runtime_seconds:.4f}",
                    result.converged,
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the optimization lab TV denoising benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory for generated figures and tables.")
    parser.add_argument("--size", type=int, default=96, help="Phantom image size.")
    parser.add_argument("--noise-sigma", type=float, default=0.08, help="Noise standard deviation.")
    parser.add_argument("--lam", type=float, default=0.08, help="TV regularization weight.")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum solver iterations.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    phantom = make_phantom(args.size)
    noisy = add_noise(phantom, sigma=args.noise_sigma)

    gd = denoise_with_gradient_descent(noisy, lam=args.lam, max_iter=args.max_iter)
    agd = denoise_with_accelerated_gradient(noisy, lam=args.lam, max_iter=args.max_iter)
    admm = denoise_with_admm(noisy, lam=args.lam, max_iter=args.max_iter)

    _save_image(args.output_dir / "phantom.png", phantom, "Synthetic phantom")
    _save_image(args.output_dir / "noisy_input.png", noisy, "Noisy input")
    _plot_benchmark(
        args.output_dir / "benchmark.png",
        phantom,
        noisy,
        [
            ("gradient_descent", gd.image),
            ("accelerated_gradient", agd.image),
            ("admm", admm.image),
        ],
    )
    _write_results_table(args.output_dir / "results.csv", phantom, [("gradient_descent", gd), ("accelerated_gradient", agd), ("admm", admm)])

    summary = {
        "methods": {
            "gradient_descent": {
                "mse": mse(phantom, gd.image),
                "psnr": psnr(phantom, gd.image),
                "iterations": gd.iterations,
                "runtime_seconds": gd.runtime_seconds,
            },
            "accelerated_gradient": {
                "mse": mse(phantom, agd.image),
                "psnr": psnr(phantom, agd.image),
                "iterations": agd.iterations,
                "runtime_seconds": agd.runtime_seconds,
            },
            "admm": {
                "mse": mse(phantom, admm.image),
                "psnr": psnr(phantom, admm.image),
                "iterations": admm.iterations,
                "runtime_seconds": admm.runtime_seconds,
            },
        }
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
