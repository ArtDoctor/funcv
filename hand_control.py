import mediapipe as mp
import numpy as np
from typing import Optional, Tuple


mp_hands = mp.solutions.hands


class HandController:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7
    ):
        self.hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.prev_pos: Optional[np.ndarray] = None
        self.movement = np.array([0.0, 0.0, 0.0])

    def process(self, rgb_frame: np.ndarray) -> Tuple[Optional[any], Optional[any]]:
        results = self.hands.process(rgb_frame)
        return results.multi_hand_landmarks, results.multi_handedness

    def update_movement(self, p1: any, p2: any, dist: float) -> None:
        if dist < 0.05:
            current = np.array([(p1.x + p2.x) / 2, (p1.y + p2.y) / 2])
            if self.prev_pos is not None:
                delta = (current - self.prev_pos) * np.array([1000, 1000])
                self.movement += np.array([delta[0], delta[1], 0])
            self.prev_pos = current
        else:
            self.prev_pos = None 