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
        approach_height: float = 0.15,
        lift_height: float = 0.25,
        grasp_threshold: float = 0.02,
        placement_height: float = 0.02,
        position_tolerance: float = 0.02,
        gripper_open_value: float = 1.0,
        gripper_close_value: float = -1.0,
        noise_scale: float = 0.0,
    ):
        """
        Initialize scripted policy.

        Args:
            approach_height: Height above object to approach from
            lift_height: Height to lift object to
            grasp_threshold: Distance threshold to trigger grasping
            placement_height: Height above target to place object
            position_tolerance: Distance tolerance for waypoint reaching
            gripper_open_value: Action value for open gripper
            gripper_close_value: Action value for closed gripper
            noise_scale: Scale of Gaussian noise to add to actions (for diversity)
        """
        self.approach_height = approach_height
        self.lift_height = lift_height
        self.grasp_threshold = grasp_threshold
        self.placement_height = placement_height
        self.position_tolerance = position_tolerance
        self.gripper_open = gripper_open_value
        self.gripper_close = gripper_close_value
        self.noise_scale = noise_scale

        self.state = PickPlaceState.APPROACH_OBJECT
        self.target_pos = None
        self.object_grasped = False

    def reset(self):
        """Reset policy state."""
        self.state = PickPlaceState.APPROACH_OBJECT
        self.target_pos = None
        self.object_grasped = False

    def _compute_waypoint_action(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        gripper_action: float,
        max_delta: float = 0.05,
    ) -> np.ndarray:
        """
        Compute action to move toward target waypoint.

        Args:
            current_pos: Current end-effector position (x, y, z)
            target_pos: Target waypoint position (x, y, z)
            gripper_action: Gripper action value
            max_delta: Maximum position delta per step

        Returns:
            Action array [dx, dy, dz, gripper] (for OSC_POSE controller)
        """
        # Compute position delta
        delta = target_pos - current_pos
        distance = np.linalg.norm(delta)

        # Clip to maximum step size
        if distance > max_delta:
            delta = delta / distance * max_delta

        # Add noise for trajectory diversity
        if self.noise_scale > 0:
            delta += np.random.randn(3) * self.noise_scale

        # Combine with gripper action
        # For OSC_POSE: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = np.zeros(7)
        action[:3] = delta
        action[6] = gripper_action

        return action

    def _reached_waypoint(self, current_pos: np.ndarray, target_pos: np.ndarray) -> bool:
        """Check if current position is close enough to target."""
        distance = np.linalg.norm(target_pos - current_pos)
        return distance < self.position_tolerance

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
                # Default target position (adjust based on environment)
                self.target_pos = np.array([0.0, 0.3, 0.82])

        # State machine
        if self.state == PickPlaceState.APPROACH_OBJECT:
            if object_pos is None:
                # Can't find object, return zero action
                return np.zeros(7), False

            # Approach from above
            target = object_pos.copy()
            target[2] += self.approach_height
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_open)

            if self._reached_waypoint(eef_pos, target):
                self.state = PickPlaceState.GRASP

        elif self.state == PickPlaceState.GRASP:
            if object_pos is None:
                return np.zeros(7), False

            # Move down to object
            target = object_pos.copy()
            target[2] += 0.02  # Slightly above object surface
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_open)

            # Check if close enough to grasp
            distance = np.linalg.norm(eef_pos - object_pos)
            if distance < self.grasp_threshold:
                self.state = PickPlaceState.LIFT
                self.object_grasped = True

        elif self.state == PickPlaceState.LIFT:
            # Lift object up
            target = eef_pos.copy()
            target[2] = self.lift_height
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_close)

            if eef_pos[2] >= self.lift_height - self.position_tolerance:
                self.state = PickPlaceState.MOVE_TO_TARGET

        elif self.state == PickPlaceState.MOVE_TO_TARGET:
            # Move to target location at lift height
            target = self.target_pos.copy()
            target[2] = self.lift_height
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_close)

            if self._reached_waypoint(eef_pos, target):
                self.state = PickPlaceState.PLACE

        elif self.state == PickPlaceState.PLACE:
            # Lower object to placement height
            target = self.target_pos.copy()
            target[2] += self.placement_height
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_close)

            if self._reached_waypoint(eef_pos, target):
                self.state = PickPlaceState.RETREAT

        elif self.state == PickPlaceState.RETREAT:
            # Open gripper and retreat
            target = eef_pos.copy()
            target[2] += 0.1
            action = self._compute_waypoint_action(eef_pos, target, self.gripper_open)

            if self._reached_waypoint(eef_pos, target):
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
