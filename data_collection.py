import cv2
import mediapipe as mp


# ── MediaPipe setup ──────────────────────────────────────────────────────────

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ── Detection & drawing ──────────────────────────────────────────────────────

def detect_landmarks(frame, model):
    """Run holistic detection on a BGR frame; returns MediaPipe results."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = model.process(rgb)
    rgb.flags.writeable = True
    return results


def draw_landmarks(frame, results):
    """Overlay all holistic landmarks on *frame* in-place."""
    # Face mesh
    mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    # Pose
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    # Left hand
    mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    # Right hand
    mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )


# ── Camera helpers ───────────────────────────────────────────────────────────

def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Open and return a VideoCapture; raises RuntimeError on failure."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {index}.")
    return cap


def read_frame(cap: cv2.VideoCapture):
    """Return (success, frame) from the capture device."""
    return cap.read()


# ── Main loop ────────────────────────────────────────────────────────────────

def run(camera_index: int = 0):
    cap = open_camera(camera_index)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            success, frame = read_frame(cap)
            if not success:
                print("Failed to grab frame — skipping.")
                continue

            results = detect_landmarks(frame, holistic)
            draw_landmarks(frame, results)

            cv2.imshow("Sign Language — Holistic Landmarks (q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
