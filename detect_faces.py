import cv2
import dlib
import numpy as np


def preprocess_frame(frame):
    """Preprocesses the given frame by converting it to grayscale, equalizing the histogram, and applying a median blur."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    gray_frame = cv2.medianBlur(gray_frame, 5)

    return gray_frame

def draw_face_mesh(frame, landmarks):
    """Draws a face mesh on the given frame using the provided set of landmarks."""
    points = np.array([[point.x, point.y] for point in landmarks.parts()])
    cv2.polylines(frame, [points], True, (0, 0, 255), 1)
    for point in landmarks.parts():
        x, y = point.x, point.y
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 1)


def detect_faces(video_source=0):
    """Detects faces in the specified video source using the Dlib library and its pre-trained models."""
    predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            preprocessed_frame = preprocess_frame(frame)
            faces = detector(frame)
            for face in faces:
                landmarks = predictor(frame, face)
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)
                draw_face_mesh(frame, landmarks)
                cv2.imshow('Face Mesh', cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error capturing frame.")
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    detect_faces()