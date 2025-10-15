"""Scripted policy for Pick Place task using state machine with waypoints."""

import numpy as np
from enum import Enum
from typing import Dict, Any, Optional, Tuple


class PickPlaceState(Enum):
    """States for the Pick Place state machine."""
    APPROACH_OBJECT = 0
    GRASP = 1
    LIFT = 2
    MOVE_TO_TARGET = 3
    PLACE = 4
    RETREAT = 5
    DONE = 6


class ScriptedPickPlacePolicy:
    """
    Scripted policy for Pick Place task using waypoint-based control.

    Uses a state machine to sequentially:
    1. Approach the object
    2. Grasp it
    3. Lift it up
    4. Move to target location
    5. Place it down
    6. Retreat
    """

    def __init__(
        self,
        approach_height: float = 0.12,  # Height above object to approach from
        lift_height: float = 1.05,  # Absolute height to lift object to (world coordinates)
        grasp_threshold: float = 0.06,  # Distance threshold to trigger grasping
        placement_height: float = 0.06,  # Height above bin to place object
        position_tolerance: float = 0.06,  # Default tolerance (slightly looser)
        position_tolerance_loose: float = 0.10,  # Looser for free space movement
        position_tolerance_tight: float = 0.04,  # Tighter for precision tasks
        gripper_open_value: float = 1.0,
        gripper_close_value: float = -1.0,
        noise_scale: float = 0.0,
        speed_multiplier: float = 2.5,  # Speed up movements (2.5x faster)
    ):
        """
        Initialize scripted policy.

        Args:
            approach_height: Height above object to approach from (relative offset)
            lift_height: Absolute world Z coordinate to lift object to (e.g., 1.05 for RoboSuite)
            grasp_threshold: Distance threshold to trigger grasping
            placement_height: Height above target to place object (relative offset)
            position_tolerance: Default distance tolerance for waypoint reaching
            position_tolerance_loose: Looser tolerance for free space movement
            position_tolerance_tight: Tighter tolerance for precision tasks
            gripper_open_value: Action value for open gripper
            gripper_close_value: Action value for closed gripper
            noise_scale: Scale of Gaussian noise to add to actions (for diversity)
            speed_multiplier: Multiplier for movement speed (1.0 = normal, 2.5 = 2.5x faster)
        """
        self.approach_height = approach_height
        self.lift_height = lift_height
        self.grasp_threshold = grasp_threshold
        self.placement_height = placement_height
        self.position_tolerance = position_tolerance
        self.position_tolerance_loose = position_tolerance_loose
        self.position_tolerance_tight = position_tolerance_tight
        self.gripper_open = gripper_open_value
        self.gripper_close = gripper_close_value
        self.noise_scale = noise_scale
        self.speed_multiplier = speed_multiplier

        self.state = PickPlaceState.APPROACH_OBJECT
        self.target_pos = None
        self.object_grasped = False
        self.stable_steps = 0  # For early success detection

    def reset(self):
        """Reset policy state."""
        self.state = PickPlaceState.APPROACH_OBJECT
        self.target_pos = None
        self.object_grasped = False
        self.stable_steps = 0

    def _compute_waypoint_action(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        gripper_action: float,
        max_delta: float = 0.05,  # Base controller output limit
        adaptive_speed: bool = True,  # Use adaptive speed based on distance
    ) -> np.ndarray:
        """
        Compute action to move toward target waypoint with adaptive speed.

        The default Panda controller expects actions in [-1, 1] range which are
        mapped to [-0.05, 0.05] meters. We normalize our deltas accordingly.

        Args:
            current_pos: Current end-effector position (x, y, z)
            target_pos: Target waypoint position (x, y, z)
            gripper_action: Gripper action value
            max_delta: Base maximum position delta per step (in meters)
            adaptive_speed: If True, move faster when far, slower when close

        Returns:
            Action array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Compute position delta
        delta = target_pos - current_pos
        distance = np.linalg.norm(delta)

        # Apply speed multiplier and adaptive speed
        effective_max_delta = max_delta * self.speed_multiplier

        if adaptive_speed:
            # Move faster when far (up to 2x), slower when very close (down to 0.5x)
            # Uses smooth sigmoid-like scaling
            if distance > 0.15:
                speed_scale = 2.0  # Fast in free space
            elif distance > 0.08:
                speed_scale = 1.5  # Medium speed
            elif distance > 0.03:
                speed_scale = 1.0  # Normal speed near target
            else:
                speed_scale = 0.5  # Slow down for precision
            effective_max_delta *= speed_scale

        # Clip to maximum step size
        if distance > effective_max_delta:
            delta = delta / distance * effective_max_delta

        # Add noise for trajectory diversity
        if self.noise_scale > 0:
            delta += np.random.randn(3) * self.noise_scale

        # Normalize delta to [-1, 1] range (controller input range)
        # Controller maps [-1, 1] to [-0.05, 0.05] meters
        normalized_delta = delta / 0.05
        # Clip to ensure within valid range
        normalized_delta = np.clip(normalized_delta, -1.0, 1.0)

        # Combine with gripper action
        # For OSC_POSE: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = np.zeros(7)
        action[:3] = normalized_delta
        action[6] = gripper_action

        return action

    def _reached_waypoint(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        Check if current position is close enough to target.

        Args:
            current_pos: Current position
            target_pos: Target position
            tolerance: Distance tolerance (uses default if None)

        Returns:
            True if within tolerance
        """
        distance = np.linalg.norm(target_pos - current_pos)
        tol = tolerance if tolerance is not None else self.position_tolerance
        return distance < tol

    def _check_early_success(
        self,
        object_pos: Optional[np.ndarray],
        target_pos: np.ndarray,
        stability_threshold: int = 15
    ) -> bool:
        """
        Check if task is already complete (object at target for several steps).

        Args:
            object_pos: Current object position
            target_pos: Target position
            stability_threshold: Number of steps object must be stable

        Returns:
            True if task appears complete
        """
        if object_pos is None:
            return False

        # Check if object is at target (tighter threshold for actual placement)
        distance = np.linalg.norm(object_pos[:2] - target_pos[:2])  # Only check x, y
        if distance < 0.04:  # Object must be very close (4cm) to target
            self.stable_steps += 1
            if self.stable_steps >= stability_threshold:
                return True
        else:
            self.stable_steps = 0

        return False

    def get_action(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """
        Get action from scripted policy based on current observation.

        Args:
            obs: Observation dict from environment wrapper

        Returns:
            action: Action array
            done: Whether episode is complete
        """
        raw_obs = obs["raw_obs"]

        # Extract end-effector position
        eef_pos = raw_obs["robot0_eef_pos"]

        # Get object and target positions (environment-specific)
        # For PickPlaceCan: object is "Can", target is placement location
        object_pos = None
        if "Can_pos" in raw_obs:
            object_pos = raw_obs["Can_pos"]
        elif "object" in raw_obs:
            object_pos = raw_obs["object"][:3]

        # Try to get target position from environment
        if self.target_pos is None:
            # For PickPlaceCan, target is typically at a fixed location
            # We'll use a default or extract from observation
            if "target" in raw_obs:
                self.target_pos = raw_obs["target"][:3]
            else:
                # RoboSuite PickPlaceCan has two bins:
                # bin1: [0.1, -0.25, 0.8]
                # bin2: [0.1, 0.28, 0.8]
                # Choose the bin closer to the can
                bin1_pos = np.array([0.1, -0.25, 0.8])
                bin2_pos = np.array([0.1, 0.28, 0.8])

                if object_pos is not None:
                    # Choose bin closest to can's initial position
                    dist1 = np.linalg.norm(object_pos[:2] - bin1_pos[:2])
                    dist2 = np.linalg.norm(object_pos[:2] - bin2_pos[:2])
                    self.target_pos = bin1_pos if dist1 < dist2 else bin2_pos
                else:
                    # Default to bin2 if we can't determine
                    self.target_pos = bin2_pos

        # Early success detection - check if task already complete
        if self._check_early_success(object_pos, self.target_pos):
            return np.zeros(7), True

        # State machine
        if self.state == PickPlaceState.APPROACH_OBJECT:
            if object_pos is None:
                # Can't find object, return zero action
                return np.zeros(7), False

            # Approach from above
            target = object_pos.copy()
            target[2] += self.approach_height

            # Concurrent action: start closing gripper when getting close
            distance_to_object = np.linalg.norm(eef_pos - object_pos)
            if distance_to_object < 0.15:
                # Close gripper while approaching
                gripper_action = self.gripper_close * min(1.0, (0.15 - distance_to_object) / 0.1)
            else:
                gripper_action = self.gripper_open

            action = self._compute_waypoint_action(eef_pos, target, gripper_action)

            # Use loose tolerance for free space movement
            if self._reached_waypoint(eef_pos, target, tolerance=self.position_tolerance_loose):
                self.state = PickPlaceState.GRASP

        elif self.state == PickPlaceState.GRASP:
            if object_pos is None:
                return np.zeros(7), False

            # Move down to object
            target = object_pos.copy()
            target[2] += 0.02  # Slightly above object surface
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_close)

            # Check if close enough to grasp (use tight tolerance)
            distance = np.linalg.norm(eef_pos - object_pos)
            if distance < self.grasp_threshold:
                self.state = PickPlaceState.LIFT
                self.object_grasped = True

        elif self.state == PickPlaceState.LIFT:
            # Lift object up
            target = eef_pos.copy()
            target[2] = self.lift_height
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_close)

            # Use default tolerance for lift
            if eef_pos[2] >= self.lift_height - self.position_tolerance:
                self.state = PickPlaceState.MOVE_TO_TARGET

        elif self.state == PickPlaceState.MOVE_TO_TARGET:
            # Move to target location at lift height (use loose tolerance for free space)
            target = self.target_pos.copy()
            target[2] = self.lift_height
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_close)

            if self._reached_waypoint(eef_pos, target, tolerance=self.position_tolerance_loose):
                self.state = PickPlaceState.PLACE

        elif self.state == PickPlaceState.PLACE:
            # Lower object to placement height
            target = self.target_pos.copy()
            target[2] += self.placement_height

            # Start opening gripper when close to placement
            if eef_pos[2] - target[2] < 0.05:
                # Gradually open gripper as we get close
                gripper_action = self.gripper_open
            else:
                gripper_action = self.gripper_close

            action = self._compute_waypoint_action(eef_pos, target, gripper_action)

            # Use tight tolerance for placement precision
            if self._reached_waypoint(eef_pos, target, tolerance=self.position_tolerance_tight):
                self.state = PickPlaceState.RETREAT

        elif self.state == PickPlaceState.RETREAT:
            # Open gripper and retreat
            target = eef_pos.copy()
            target[2] += 0.1
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_open)

            # Use loose tolerance for retreat
            if self._reached_waypoint(eef_pos, target, tolerance=self.position_tolerance_loose):
                self.state = PickPlaceState.DONE

        elif self.state == PickPlaceState.DONE:
            action = np.zeros(7)
            return action, True

        else:
            action = np.zeros(7)

        return action, False

    def get_state_name(self) -> str:
        """Get current state name for debugging."""
        return self.state.name
