"""RoboSuite environment wrapper for ACT training."""

import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from typing import Dict, Any, Optional, List


class RoboSuiteWrapper:
    """Wrapper for RoboSuite environments with standardized observation format."""

    def __init__(
        self,
        env_name: str = "PickPlaceCan",
        robots: str = "Panda",
        controller_name: str = "OSC_POSE",
        camera_names: Optional[List[str]] = None,
        camera_height: int = 84,
        camera_width: int = 84,
        horizon: int = 500,
        reward_shaping: bool = True,
    ):
        """
        Initialize RoboSuite environment wrapper.

        Args:
            env_name: Name of the RoboSuite task (e.g., "PickPlaceCan")
            robots: Robot type (e.g., "Panda", "Sawyer")
            controller_name: Controller type (e.g., "OSC_POSE" for end-effector control)
            camera_names: List of camera names to use for observations
            camera_height: Height of camera images
            camera_width: Width of camera images
            horizon: Maximum episode length
            reward_shaping: Whether to use shaped rewards
        """
        self.env_name = env_name
        self.robots = robots
        self.controller_name = controller_name
        self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
        self.camera_height = camera_height
        self.camera_width = camera_width

        # Load controller config
        controller_config = load_controller_config(default_controller=controller_name)

        # Create environment
        self.env = suite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.camera_names,
            camera_heights=camera_height,
            camera_widths=camera_width,
            horizon=horizon,
            reward_shaping=reward_shaping,
        )

        self.action_dim = self.env.action_spec[0].shape[0]
        self.state_dim = self._get_state_dim()

    def _get_state_dim(self) -> int:
        """Get dimensionality of proprioceptive state."""
        obs = self.env.reset()
        state = self._extract_state(obs)
        return state.shape[0]

    def _extract_state(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract proprioceptive state from observation dict.

        Includes robot joint positions, velocities, gripper state, and object poses.
        """
        state_components = []

        # Robot joint positions and velocities
        if "robot0_joint_pos" in obs:
            state_components.append(obs["robot0_joint_pos"])
        if "robot0_joint_vel" in obs:
            state_components.append(obs["robot0_joint_vel"])

        # End-effector pose
        if "robot0_eef_pos" in obs:
            state_components.append(obs["robot0_eef_pos"])
        if "robot0_eef_quat" in obs:
            state_components.append(obs["robot0_eef_quat"])

        # Gripper state
        if "robot0_gripper_qpos" in obs:
            state_components.append(obs["robot0_gripper_qpos"])
        if "robot0_gripper_qvel" in obs:
            state_components.append(obs["robot0_gripper_qvel"])

        # Object poses (if available)
        for key in obs.keys():
            if "_pos" in key and "robot0" not in key:
                state_components.append(obs[key])
            if "_quat" in key and "robot0" not in key:
                state_components.append(obs[key])

        return np.concatenate(state_components)

    def _extract_images(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract camera images from observation dict."""
        images = {}
        for cam_name in self.camera_names:
            key = f"{cam_name}_image"
            if key in obs:
                # Convert from (H, W, C) to (C, H, W) for PyTorch
                images[cam_name] = np.transpose(obs[key], (2, 0, 1))
        return images

    def reset(self) -> Dict[str, Any]:
        """
        Reset environment and return initial observation.

        Returns:
            Dictionary with 'state' (proprioceptive) and 'images' (camera observations)
        """
        obs = self.env.reset()
        return {
            "state": self._extract_state(obs),
            "images": self._extract_images(obs),
            "raw_obs": obs,
        }

    def step(self, action: np.ndarray) -> tuple:
        """
        Take environment step with action.

        Args:
            action: Action array

        Returns:
            obs: Observation dict with 'state' and 'images'
            reward: Scalar reward
            done: Boolean done flag
            info: Additional information dict
        """
        obs, reward, done, info = self.env.step(action)

        obs_dict = {
            "state": self._extract_state(obs),
            "images": self._extract_images(obs),
            "raw_obs": obs,
        }

        return obs_dict, reward, done, info

    def get_object_pose(self, obs: Dict[str, np.ndarray], object_name: str) -> Optional[np.ndarray]:
        """Get pose of a specific object from raw observations."""
        pos_key = f"{object_name}_pos"
        quat_key = f"{object_name}_quat"

        raw_obs = obs.get("raw_obs", {})
        if pos_key in raw_obs:
            pose = [raw_obs[pos_key]]
            if quat_key in raw_obs:
                pose.append(raw_obs[quat_key])
            return np.concatenate(pose)
        return None

    def close(self):
        """Close the environment."""
        self.env.close()
