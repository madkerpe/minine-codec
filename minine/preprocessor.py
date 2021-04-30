import cv2
import dlib


class Preprocessor:
    def __init__(self, detection_refresh_rate=10, padding=50):
        self.detector = dlib.get_frontal_face_detector()
        self.padding = padding
        self.detection_refresh_rate = detection_refresh_rate
        self.detection_refresh_count = 0
        self.last_rectangle = None

    def detect_face(self, frame):
        """
        Return a dlib::rectangle around every face in the frame
        """
        if (
            self.last_rectangle == None
            or self.detection_refresh_count > self.detection_refresh_rate
        ):
            self.detection_refresh_count = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            self.last_rectangle = rects
        else:
            self.detection_refresh_count += 1
            rects = self.last_rectangle

        return rects

    def cut_face(self, frame, rect):
        p = self.padding
        frame = frame[
            rect.top() - p : rect.bottom() + p, rect.left() - p : rect.right() + p
        ]
        return frame

    def resize_image(self, frame, size=(256, 256)):
        frame = cv2.resize(frame, size)
        return frame
