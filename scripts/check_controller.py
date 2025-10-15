"""Check controller configuration and available options."""

import sys
sys.path.insert(0, '.')

import robosuite as suite
import json


def check_controller_config():
    """Examine controller configuration."""
    print("="*60)
    print("CHECKING CONTROLLER CONFIGURATION")
    print("="*60)

    env = suite.make(
        env_name="PickPlaceCan",
        robots="Panda",
        has_renderer=False,
        use_camera_obs=False,
        horizon=500,
        control_freq=50,
    )

    print(f"\nController name: {env.robots[0].controller.name}")
    print(f"Controller type: {type(env.robots[0].controller)}")

    if hasattr(env.robots[0].controller, 'gripper_controller'):
        gripper_ctrl = env.robots[0].controller.gripper_controller
        print(f"\nGripper controller: {gripper_ctrl}")
        print(f"Gripper controller type: {type(gripper_ctrl)}")

    # Check gripper
    gripper = env.robots[0].gripper
    print(f"\nGripper: {gripper}")
    print(f"Gripper type: {type(gripper)}")
    print(f"Gripper dof: {gripper.dof}")

    # Check action limits
    print(f"\nAction spec: {env.action_spec}")
    print(f"Action dim: {env.action_dim}")

    # Try different controllers
    print("\n" + "="*60)
    print("TESTING WITH DIFFERENT CONTROLLER")
    print("="*60)

    from robosuite.controllers import load_composite_controller_config

    # Try with JOINT_POSITION controller
    controller_config = load_composite_controller_config(
        controller="OSC_POSE",
        robot="Panda",
    )

    print(f"\nController config:")
    print(json.dumps(controller_config, indent=2, default=str))

    env.close()

    # Try env with explicit gripper control
    print("\n" + "="*60)
    print("CHECKING GRIPPER CONTROL MODE")
    print("="*60)

    env = suite.make(
        env_name="PickPlaceCan",
        robots="Panda",
        has_renderer=False,
        use_camera_obs=False,
        horizon=500,
        control_freq=50,
    )

    obs = env.reset()

    # Check if gripper has special control modes
    gripper = env.robots[0].gripper
    print(f"Gripper actuators: {gripper.important_actuators}")
    print(f"Gripper joints: {gripper.important_joints}")

    # Try to see gripper force
    import numpy as np
    for i in range(20):
        action = np.zeros(7)
        action[6] = 1.0  # Close
        obs, _, _, _ = env.step(action)

    # Check contact forces
    if hasattr(env.sim.data, 'contact'):
        print(f"\nNumber of contacts: {env.sim.data.ncon}")
        for i in range(min(5, env.sim.data.ncon)):
            contact = env.sim.data.contact[i]
            body1 = env.sim.model.body_id2name(env.sim.model.geom_bodyid[contact.geom1])
            body2 = env.sim.model.body_id2name(env.sim.model.geom_bodyid[contact.geom2])
            print(f"  Contact {i}: {body1} <-> {body2}, dist={contact.dist:.4f}")

    env.close()


if __name__ == "__main__":
    check_controller_config()
