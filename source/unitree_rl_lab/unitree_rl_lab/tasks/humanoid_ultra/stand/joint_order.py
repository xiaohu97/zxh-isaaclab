"""Canonical Isaac Lab joint order for Humanoid Ultra 27-DoF policies."""

HUMANOID_ULTRA_27DOF_JOINT_ORDER = (
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_yaw_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
)


def validate_humanoid_ultra_27dof_joint_order(actual_joint_names: list[str] | tuple[str, ...]) -> None:
    """Fail before training if Isaac Lab exposes a different policy/action order."""
    actual = tuple(actual_joint_names)
    expected = HUMANOID_ULTRA_27DOF_JOINT_ORDER
    if actual == expected:
        return

    mismatch_lines = []
    for index in range(max(len(actual), len(expected))):
        expected_name = expected[index] if index < len(expected) else "<missing>"
        actual_name = actual[index] if index < len(actual) else "<missing>"
        if expected_name != actual_name:
            mismatch_lines.append(
                f"  index {index:02d}: expected {expected_name}, got {actual_name}"
            )
    raise RuntimeError(
        "Humanoid Ultra 27-DoF joint order does not match the policy mapping:\n"
        + "\n".join(mismatch_lines)
    )
