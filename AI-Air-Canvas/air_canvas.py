"""
AI Air Canvas - Real-Time Writing Using Hand Gestures
=====================================================
Main application entry point.
Uses webcam + MediaPipe to track hand landmarks and draw on a virtual canvas.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# ── Import custom hand tracking utility ──────────────────────────────────────
from utils.hand_tracking import HandTracker

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_NAME   = "✨ AI Air Canvas"
HEADER_HEIGHT = 80          # Height of the top colour-selector bar (px)
BRUSH_RADIUS  = 6           # Drawing brush thickness
SMOOTHING     = 8           # Deque length for trajectory smoothing

# Colour palette  (BGR for OpenCV)
COLOURS = {
    "Red":    (0,   0,   220),
    "Green":  (0,   180, 60),
    "Blue":   (220, 80,  0),
    "Yellow": (0,   220, 220),
}
CLEAR_LABEL = "Clear"

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER / BUTTON BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_header(width: int, active_colour: str) -> np.ndarray:
    """
    Draw a soft-pink header bar with labelled colour buttons.
    Returns an image of shape (HEADER_HEIGHT, width, 3).
    """
    header = np.full((HEADER_HEIGHT, width, 3), (230, 210, 220), dtype=np.uint8)

    # Subtle gradient tint
    for y in range(HEADER_HEIGHT):
        alpha = y / HEADER_HEIGHT
        header[y] = np.clip(
            header[y] * (1 - alpha * 0.15) + np.array([255, 235, 245]) * alpha * 0.15,
            0, 255
        ).astype(np.uint8)

    # Title text
    cv2.putText(header, "AI Air Canvas", (14, 52),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (180, 100, 140), 2, cv2.LINE_AA)

    # Buttons
    buttons = list(COLOURS.keys()) + [CLEAR_LABEL]
    btn_w   = 100
    gap     = 14
    start_x = width - (btn_w + gap) * len(buttons) - gap

    button_rects = {}
    for i, label in enumerate(buttons):
        x1 = start_x + i * (btn_w + gap)
        x2 = x1 + btn_w
        y1, y2 = 12, HEADER_HEIGHT - 12

        # Highlight active colour
        is_active = (label == active_colour)
        if label == CLEAR_LABEL:
            fill = (200, 180, 210)
            border = (160, 120, 170)
        else:
            fill   = COLOURS[label]
            border = tuple(max(0, c - 60) for c in fill)

        # Draw rounded button body
        cv2.rectangle(header, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(header, (x1, y1), (x2, y2),
                      border if not is_active else (255, 255, 255), 3 if is_active else 1)

        # Shine strip at top
        cv2.rectangle(header, (x1 + 2, y1 + 2), (x2 - 2, y1 + 10),
                      tuple(min(255, c + 60) for c in fill), -1)

        # Label
        txt_col = (255, 255, 255) if label != CLEAR_LABEL else (80, 40, 80)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(header, label,
                    (x1 + (btn_w - tw) // 2, y1 + (y2 - y1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_col, 1, cv2.LINE_AA)

        button_rects[label] = (x1, y1, x2, y2)

    return header, button_rects


# ─────────────────────────────────────────────────────────────────────────────
#  GESTURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def count_raised_fingers(landmarks, hand_label: str = "Right") -> int:
    """
    Count how many fingers are raised.
    Uses tip-vs-pip comparison for index→pinky; special logic for thumb.
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    count = 0

    # Thumb (horizontal comparison flipped for left hand)
    if hand_label == "Right":
        if landmarks[tips[0]].x < landmarks[pips[0]].x:
            count += 1
    else:
        if landmarks[tips[0]].x > landmarks[pips[0]].x:
            count += 1

    # Index → Pinky: tip above pip means raised
    for tip, pip in zip(tips[1:], pips[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Init webcam ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check device index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read from webcam.")
        return

    h, w = frame.shape[:2]

    # ── Init drawing canvas (transparent pink-white base) ───────────────────
    canvas = np.zeros((h, w, 3), dtype=np.uint8)  # black = transparent layer

    # ── Init hand tracker ───────────────────────────────────────────────────
    tracker = HandTracker(max_hands=1, detection_confidence=0.75, tracking_confidence=0.75)

    # ── State variables ──────────────────────────────────────────────────────
    active_colour  = "Red"
    prev_point     = None
    smooth_pts     = deque(maxlen=SMOOTHING)   # smoothing buffer

    print("[INFO] AI Air Canvas started. Press 'Q' or ESC to quit.")
    print("[INFO] ✌️  2+ fingers raised → STOP drawing  |  ☝️ 1 finger → DRAW")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror (selfie view)
        frame = cv2.flip(frame, 1)

        # ── Hand detection ───────────────────────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = tracker.find_hands(rgb_frame)

        index_tip  = None
        drawing    = False
        hand_label = "Right"

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = hand_info.classification[0].label

                # Get index fingertip pixel coords
                lm = hand_landmarks.landmark
                ix = int(lm[8].x * w)
                iy = int(lm[8].y * h)
                index_tip = (ix, iy)

                # Count raised fingers
                fingers_up = count_raised_fingers(lm, hand_label)

                # ── Finger logic ─────────────────────────────────────────────
                # Only index finger raised → draw
                if fingers_up == 1:
                    drawing = True

                # 2+ fingers → stop drawing / hover mode
                else:
                    drawing   = False
                    prev_point = None

                    # Check if hovering over a header button
                    if iy < HEADER_HEIGHT:
                        for label, (x1, y1, x2, y2) in button_rects.items():
                            if x1 <= ix <= x2:
                                if label == CLEAR_LABEL:
                                    canvas[:] = 0   # wipe canvas
                                else:
                                    active_colour = label

                # Draw landmark skeleton
                tracker.draw_landmarks(frame, hand_landmarks)

                # Fingertip cursor dot
                cv2.circle(frame, index_tip, 10,
                           COLOURS.get(active_colour, (255, 255, 255)), -1)
                cv2.circle(frame, index_tip, 12, (255, 255, 255), 2)

        # ── Smooth & draw on canvas ──────────────────────────────────────────
        if drawing and index_tip:
            smooth_pts.append(index_tip)

            # Average recent points for smoother stroke
            sx = int(np.mean([p[0] for p in smooth_pts]))
            sy = int(np.mean([p[1] for p in smooth_pts]))
            smooth_point = (sx, sy)

            if prev_point and sy > HEADER_HEIGHT:
                cv2.line(canvas, prev_point, smooth_point,
                         COLOURS[active_colour], BRUSH_RADIUS * 2)
                cv2.line(canvas, prev_point, smooth_point,
                         tuple(min(255, c + 40) for c in COLOURS[active_colour]),
                         BRUSH_RADIUS)  # lighter inner stroke = glow effect

            prev_point = smooth_point
        else:
            prev_point = None
            smooth_pts.clear()

        # ── Composite canvas onto frame ──────────────────────────────────────
        # Where canvas has non-zero pixels, overlay with 85% opacity
        canvas_mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask     = cv2.threshold(canvas_mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv    = cv2.bitwise_not(mask)

        frame_bg    = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg   = cv2.addWeighted(cv2.bitwise_and(canvas, canvas, mask=mask),
                                       0.85, frame_bg * 0, 0.0, 0)
        frame       = cv2.add(frame_bg, canvas_fg)

        # ── Build and overlay header ─────────────────────────────────────────
        header, button_rects = build_header(w, active_colour)
        frame[:HEADER_HEIGHT] = cv2.addWeighted(
            frame[:HEADER_HEIGHT], 0.25, header, 0.75, 0
        )

        # ── Status text ──────────────────────────────────────────────────────
        status = "✏ DRAWING" if drawing else "✋ HOVER"
        col    = COLOURS.get(active_colour, (200, 200, 200))
        cv2.putText(frame, status, (14, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
        cv2.putText(frame, "Q / ESC = Quit", (w - 170, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 130, 160), 1, cv2.LINE_AA)

        # ── Show window ──────────────────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q or ESC
            break
        elif key == ord('c'):                  # keyboard shortcut to clear
            canvas[:] = 0

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] AI Air Canvas closed.")


if __name__ == "__main__":
    main()
