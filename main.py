import cv2
import numpy as np
import math
from geometry_utils import create_square, rotate_y, project
from hand_control import HandController

# State
angle: float = 0
rotation_speed: float = 0.02
base_size: float = 100
movement: np.ndarray = np.array([0.0, 0.0, 0.0])

# Interpolated state
interp_angle: float = 0
interp_rotation_speed: float = 0.02
interp_base_size: float = 100
interp_movement: np.ndarray = np.array([0.0, 0.0, 0.0])

# Init hand controller
hand_controller = HandController()

# Init webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 120)


def lerp(a, b, t):
    return a + (b - a) * t


def draw_3d_circle(
    output: np.ndarray,
    center3d: np.ndarray,
    radius: float,
    angle: float,
    movement: np.ndarray,
    color: tuple,
    thickness: int = 2,
    num_points: int = 100,
    orientation: float = 0.0
):
    # Create points in 3D circle in XZ plane, then rotate around Y axis
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.stack([
        np.cos(theta) * radius,
        np.zeros_like(theta),
        np.sin(theta) * radius
    ], axis=1)
    # Rotate circle in 3D (orientation around X axis for tilt, angle for spinning)
    # First, rotate around X for tilt
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(orientation), -np.sin(orientation)],
        [0, np.sin(orientation), np.cos(orientation)]
    ])
    # Then, rotate around Y for spinning
    Ry = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    rotated = circle_points @ Rx.T @ Ry.T + center3d
    # Project to 2D
    projected = project(rotated, movement)
    pts_2d = projected[:, :2].astype(np.int32)
    cv2.polylines(output, [pts_2d], isClosed=True, color=color, thickness=thickness)


def main() -> None:
    global angle, rotation_speed, base_size, movement
    global interp_angle, interp_rotation_speed, interp_base_size, interp_movement
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        multi_hand_landmarks, multi_handedness = hand_controller.process(rgb)
        output = np.zeros((480, 640, 3), dtype=np.uint8)

        finger_line_drawn = False

        if multi_hand_landmarks and multi_handedness:
            for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
                label = handedness.classification[0].label
                landmarks = hand_landmarks.landmark

                p1, p2 = landmarks[4], landmarks[8]
                x1, y1 = int(p1.x * 640), int(p1.y * 480)
                x2, y2 = int(p2.x * 640), int(p2.y * 480)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                finger_line_drawn = True

                dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

                if label == 'Left':
                    rotation_speed = 0.01 + dist * 0.2
                elif label == 'Right':
                    base_size = 30 + dist * 300
                    hand_controller.update_movement(p1, p2, dist)
                    movement = hand_controller.movement

        # Interpolate parameters for smooth transitions
        interp_rotation_speed = lerp(interp_rotation_speed, rotation_speed, 0.15)
        interp_base_size = lerp(interp_base_size, base_size, 0.15)
        interp_movement = lerp(interp_movement, movement, 0.15)
        interp_angle += interp_rotation_speed

        all_quads = []

        # 3 concentric rings of squares
        for layer in range(3):
            ring_radius = 150 + layer * 120
            size = interp_base_size * (1.0 + 0.4 * layer)
            base_square = create_square(size)

            for i in range(4):
                theta = interp_angle + i * (np.pi / 2)
                offset = np.array([
                    ring_radius * np.cos(theta),
                    0,
                    ring_radius * np.sin(theta)
                ])
                # Compute direction to center and align square to it
                dir_to_center = -offset / np.linalg.norm(offset)
                angle_to_center = np.arctan2(dir_to_center[2], dir_to_center[0])  # angle in XZ plane
                facing_square = rotate_y(base_square, - angle_to_center / 2 + np.pi / 6)
                rotated = facing_square + offset

                projected = project(rotated, interp_movement)

                # Z-depth for sorting (use average Z)
                avg_z = np.mean(projected[:, 2])
                pts_2d = projected[:, :2].astype(np.int32)
                all_quads.append((avg_z, pts_2d))

        # Depth sort: farthest first
        all_quads.sort(key=lambda x: -x[0])

        for _, pts in all_quads:
            cv2.fillPoly(output, [pts], color=(80, 80, 80))      # Gray fill
            cv2.polylines(output, [pts], isClosed=True, color=(255, 255, 255), thickness=2)  # White border

        # Draw 3D spinning circles centered at the origin, with orientation
        for i, (radius, color, orient) in enumerate([
            (180, (0, 255, 255), interp_angle * 0.7),
            (260, (255, 100, 100), interp_angle * 1.1),
            (340, (100, 255, 100), interp_angle * 1.7)
        ]):
            draw_3d_circle(
                output,
                center3d=np.array([0, 0, 0]),
                radius=radius,
                angle=interp_angle * (1.0 + 0.2 * i),
                movement=interp_movement,
                color=color,
                thickness=2,
                orientation=orient
            )

        cv2.imshow('Webcam', frame if finger_line_drawn else np.zeros_like(frame))
        cv2.imshow('3D Rotating Structure', output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
