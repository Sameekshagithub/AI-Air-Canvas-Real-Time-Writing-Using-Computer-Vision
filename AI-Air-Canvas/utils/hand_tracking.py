"""
utils/hand_tracking.py
======================
Thin wrapper around MediaPipe Hands for clean, reusable hand landmark detection.
"""


import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    """
    Wraps MediaPipe Hands to provide:
      - find_hands(rgb_frame)  → raw MediaPipe results
      - draw_landmarks(frame, hand_landmarks)  → draws skeleton on BGR frame
    """

    # MediaPipe drawing utilities (shared across all instances)
    _mp_draw   = mp.solutions.drawing_utils
    _mp_styles = mp.solutions.drawing_styles
    _mp_hands  = mp.solutions.hands

    # Custom landmark drawing style (soft pink / white)
    _LANDMARK_STYLE = _mp_draw.DrawingSpec(
        color=(200, 150, 180), thickness=2, circle_radius=3
    )
    _CONNECTION_STYLE = _mp_draw.DrawingSpec(
        color=(240, 200, 220), thickness=2
    )

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.75,
        tracking_confidence:  float = 0.75,
    ):
        """
        Args:
            max_hands:             Maximum number of hands to track simultaneously.
            detection_confidence:  Minimum confidence for initial detection.
            tracking_confidence:   Minimum confidence for landmark tracking.
        """
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def find_hands(self, rgb_frame):
        """
        Run MediaPipe inference on an RGB frame.

        Args:
            rgb_frame: numpy array in RGB colour order.

        Returns:
            MediaPipe results object  (results.multi_hand_landmarks,
                                       results.multi_handedness)
        """
        # Mark as non-writeable to improve performance
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results

    # ─────────────────────────────────────────────────────────────────────────
    def draw_landmarks(self, bgr_frame, hand_landmarks):
        """
        Overlay MediaPipe hand skeleton on a BGR frame (in-place).

        Args:
            bgr_frame:      OpenCV BGR image to draw on.
            hand_landmarks: A single hand's landmark object from results.
        """
        self._mp_draw.draw_landmarks(
            bgr_frame,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._LANDMARK_STYLE,
            self._CONNECTION_STYLE,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def get_landmark_px(self, hand_landmarks, index: int, width: int, height: int):
        """
        Convert a normalised landmark to pixel coordinates.

        Args:
            hand_landmarks: MediaPipe hand landmarks object.
            index:          Landmark index (0-20).
            width:          Frame width in pixels.
            height:         Frame height in pixels.

        Returns:
            (x_px, y_px) tuple.
        """
        lm = hand_landmarks.landmark[index]
        return int(lm.x * width), int(lm.y * height)

    # ─────────────────────────────────────────────────────────────────────────
    def __del__(self):
        """Release MediaPipe resources on garbage collection."""
        try:
            self._hands.close()
        except Exception:
            pass
