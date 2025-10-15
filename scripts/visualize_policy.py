"""Visualize scripted policy execution."""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from act_implementation.envs.scripted_policy import ScriptedPickPlacePolicy


def visualize_policy(
    env_name="PickPlaceCan",
    robot="Panda",
    episodes=3,
    noise=0.003,
    output_video=None,
    live=False,
):
    """Visualize scripted policy with live rendering or video recording.

    Args:
        env_name: Environment name
        robot: Robot type
        episodes: Number of episodes to run
        noise: Action noise scale
        output_video: Output video file path (if None and not live, uses policy_viz.mp4)
        live: If True, show live rendering window. If False, record video.
    """
    if not live and output_video is None:
        output_video = "policy_viz.mp4"

    if not live:
        import imageio

    # Create environment with appropriate rendering mode
    env = suite.make(
        env_name=env_name,
        robots=robot,
        has_renderer=live,
        has_offscreen_renderer=not live,
        use_camera_obs=not live,
        camera_names=["frontview"] if not live else None,
        camera_heights=512 if not live else None,
        camera_widths=512 if not live else None,
        horizon=500,
        control_freq=50,
    )

    policy = ScriptedPickPlacePolicy(noise_scale=noise)

    all_frames = [] if not live else None

    for ep in range(episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{episodes}")
        print(f"{'='*60}")

        obs = env.reset()
        policy.reset()

        # Get initial positions
        eef_pos = obs["robot0_eef_pos"]
        can_pos = obs["Can_pos"]

        print(f"Initial EEF position: {eef_pos}")
        print(f"Can position: {can_pos}")
        print(f"Target position: {policy.target_pos}")

        for step in range(500):
            # Get policy action
            obs_dict = {
                "state": None,
                "images": None,
                "raw_obs": obs,
            }
            action, done = policy.get_action(obs_dict)

            # Take step
            obs, reward, env_done, info = env.step(action)

            # Render or record frame
            if live:
                env.render()
            else:
                frame = obs["frontview_image"][::-1]  # Flip vertically
                all_frames.append(frame)

            # Print progress every 50 steps
            if step % 50 == 0:
                eef_pos = obs["robot0_eef_pos"]
                can_pos = obs["Can_pos"]
                print(f"Step {step:3d}: State={policy.get_state_name():20s} "
                      f"Reward={reward:.3f} EEF={eef_pos}")

            if done or env_done:
                success = info.get("success", False)
                print(f"\nEpisode finished at step {step}")
                print(f"Success: {success}")
                print(f"Final state: {policy.get_state_name()}")
                print(f"Final reward: {reward:.3f}")
                break

        if not (done or env_done):
            print(f"\nEpisode TIMEOUT at state {policy.get_state_name()}")

    env.close()

    # Save video if not live
    if not live:
        print(f"\nSaving video with {len(all_frames)} frames to {output_video}...")
        imageio.mimsave(output_video, all_frames, fps=20)
        print(f"Video saved! Open {output_video} to view.")
    else:
        print("\nLive visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize scripted policy")
    parser.add_argument("--env", type=str, default="PickPlaceCan", help="Environment name")
    parser.add_argument("--robot", type=str, default="Panda", help="Robot type")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--noise", type=float, default=0.003, help="Action noise scale")
    parser.add_argument("--output", type=str, default=None, help="Output video file (only for non-live mode)")
    parser.add_argument("--live", action="store_true", help="Show live rendering window instead of recording video")
    args = parser.parse_args()

    visualize_policy(
        env_name=args.env,
        robot=args.robot,
        episodes=args.episodes,
        noise=args.noise,
        output_video=args.output,
        live=args.live,
    )
