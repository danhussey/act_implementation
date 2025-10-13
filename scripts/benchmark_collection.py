"""Benchmark data collection speed with different configurations."""

import argparse
import time
import subprocess
from pathlib import Path
import tempfile
import shutil


def run_collection(
    script: str,
    episodes: int,
    workers: int = None,
    resolution: str = "84x84",
    output_dir: Path = None
) -> float:
    """
    Run data collection and measure time.

    Args:
        script: Script name ('serial' or 'parallel')
        episodes: Number of episodes to collect
        workers: Number of workers (for parallel only)
        resolution: Image resolution (e.g., "84x84", "480x640")
        output_dir: Output directory for data

    Returns:
        Time taken in seconds
    """
    height, width = map(int, resolution.split('x'))

    # Create temporary output file
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))

    output_file = output_dir / f"{script}_{episodes}ep_{resolution}.hdf5"

    # Build command
    if script == "serial":
        cmd = [
            "uv", "run", "python", "collect_demos.py",
            "--episodes", str(episodes),
            "--output", str(output_file),
            "--camera-height", str(height),
            "--camera-width", str(width),
        ]
    else:  # parallel
        cmd = [
            "uv", "run", "python", "collect_demos_parallel.py",
            "--episodes", str(episodes),
            "--workers", str(workers or 4),
            "--output", str(output_file),
            "--camera-height", str(height),
            "--camera-width", str(width),
        ]

    print(f"\n{'='*60}")
    print(f"Running: {script.upper()} collection")
    print(f"  Episodes: {episodes}")
    if script == "parallel":
        print(f"  Workers: {workers or 4}")
    print(f"  Resolution: {resolution}")
    print(f"{'='*60}\n")

    # Run and time
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
        return elapsed
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Collection failed: {e}")
        return -1
    except KeyboardInterrupt:
        print(f"\n✗ Interrupted by user")
        return -1


def main():
    parser = argparse.ArgumentParser(description="Benchmark collection speed")
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to collect per test"
    )
    parser.add_argument(
        "--test-parallel", action="store_true",
        help="Test parallel collection with different worker counts"
    )
    parser.add_argument(
        "--test-resolution", action="store_true",
        help="Test different resolutions"
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[2, 4, 8],
        help="Worker counts to test (for --test-parallel)"
    )
    parser.add_argument(
        "--resolutions", type=str, nargs="+", default=["84x84", "256x256", "480x640"],
        help="Resolutions to test (for --test-resolution)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: temp dir)"
    )
    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))

    print(f"Benchmark Configuration:")
    print(f"  Episodes per test: {args.episodes}")
    print(f"  Output directory: {output_dir}")

    results = []

    try:
        # Baseline: serial collection
        print("\n" + "="*60)
        print("BASELINE: Serial Collection")
        print("="*60)
        time_serial = run_collection(
            "serial", args.episodes, output_dir=output_dir
        )
        if time_serial > 0:
            results.append(("Serial (84x84)", time_serial, 1.0))

        # Test parallel with different worker counts
        if args.test_parallel:
            print("\n" + "="*60)
            print("TEST: Parallel Collection (Different Worker Counts)")
            print("="*60)

            for workers in args.workers:
                time_parallel = run_collection(
                    "parallel", args.episodes, workers=workers, output_dir=output_dir
                )
                if time_parallel > 0 and time_serial > 0:
                    speedup = time_serial / time_parallel
                    results.append((f"Parallel ({workers} workers)", time_parallel, speedup))

        # Test different resolutions (parallel only, 4 workers)
        if args.test_resolution:
            print("\n" + "="*60)
            print("TEST: Different Resolutions (4 workers)")
            print("="*60)

            for resolution in args.resolutions:
                time_res = run_collection(
                    "parallel", args.episodes, workers=4,
                    resolution=resolution, output_dir=output_dir
                )
                if time_res > 0:
                    results.append((f"Parallel 4w ({resolution})", time_res, None))

        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"{'Configuration':<30} {'Time (min)':<12} {'Speedup':<10}")
        print("-" * 60)

        for config, time_taken, speedup in results:
            time_min = time_taken / 60
            speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
            print(f"{config:<30} {time_min:<12.2f} {speedup_str:<10}")

        print("="*60)

        # Estimate time for full collection
        if time_serial > 0:
            print("\nEstimated time for 100 episodes:")
            time_per_episode = time_serial / args.episodes
            print(f"  Serial: {time_per_episode * 100 / 60:.1f} minutes ({time_per_episode * 100 / 3600:.2f} hours)")

            for config, time_taken, speedup in results:
                if "Parallel" in config and speedup:
                    time_per_episode_parallel = time_taken / args.episodes
                    print(f"  {config}: {time_per_episode_parallel * 100 / 60:.1f} minutes ({time_per_episode_parallel * 100 / 3600:.2f} hours)")

    finally:
        # Cleanup
        if not args.output_dir:
            print(f"\nCleaning up temporary files in {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
