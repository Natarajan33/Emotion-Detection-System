# import cv2
# from deepface import DeepFace


# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# cap = cv2.VideoCapture(1)

# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")
# while True:
#     ret,frame = cap.read()
#     result = DeepFace.analyze(frame, actions = ['emotion'])

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray,1.1,4)

#     for(x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

#     font = cv2.FONT_HERSHEY_SIMPLEX  

#     cv2.putText(frame,
#                 result[0]['dominant_emotion'],
#                 (50,50),
#                 font,3,
#                 (0,255,0),
#                 2,
#                 cv2.LINE_4)
#     cv2.imshow('Demo Video',frame)

#     if cv2.waitKey(2) & 0XFF == ord('q'):
#          break
    
# cap.release()
# cv2.destroyAllWindows()    
      
import cv2
from deepface import DeepFace
import numpy as np
from typing import Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    def _init_(self):
        # Initialize the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            raise ValueError("Error loading face cascade classifier")

        # Initialize variables for FPS calculation
        self.fps_start_time = time.time()
        self.fps = 0
        self.frame_count = 0

        # Initialize emotion detection parameters
        self.last_emotion = None
        self.emotion_confidence = 0
        self.skip_frames = 0
        self.process_this_frame = True
        
        # Initialize ThreadPoolExecutor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def get_camera(self) -> Optional[cv2.VideoCapture]:
        """Try to get camera access, attempting multiple devices"""
        for device_index in range(2):  # Try first two camera devices
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                # Set optimal camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
        
        raise IOError("No accessible webcam found")

    def process_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """Process frame to detect emotion using DeepFace"""
        try:
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            if result:
                return (
                    result[0]['dominant_emotion'],
                    max(result[0]['emotion'].values())
                )
        except Exception as e:
            logger.warning(f"Error in emotion detection: {str(e)}")
        return (self.last_emotion if self.last_emotion else "Unknown", 0.0)

    def draw_info(self, frame: np.ndarray, emotion: str, confidence: float):
        """Draw emotion and performance information on frame"""
        # Calculate FPS
        self.frame_count += 1
        if time.time() - self.fps_start_time >= 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = time.time()

        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 110), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw emotion and confidence
        cv2.putText(frame, f"Emotion: {emotion}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def detect_and_draw_faces(self, frame: np.ndarray) -> np.ndarray:
        """Detect and draw rectangles around faces"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw emotion indicator bar
            if self.last_emotion and self.emotion_confidence > 0:
                bar_width = int(w * self.emotion_confidence)
                cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+10),
                            (0, 255, 0), -1)

        return frame

    def run(self):
        """Main loop for emotion detection"""
        try:
            cap = self.get_camera()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break

                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 480))

                # Process emotion detection every N frames
                if self.process_this_frame:
                    if self.future is None or self.future.done():
                        self.future = self.executor.submit(self.process_frame, frame)
                
                if self.future and self.future.done():
                    self.last_emotion, self.emotion_confidence = self.future.result()
                
                # Update frame processing flag
                self.skip_frames = (self.skip_frames + 1) % 3  # Process every 3rd frame
                self.process_this_frame = self.skip_frames == 0

                # Detect and draw faces
                frame = self.detect_and_draw_faces(frame)
                
                # Draw information on frame
                frame = self.draw_info(frame, 
                                     self.last_emotion if self.last_emotion else "Unknown",
                                     self.emotion_confidence)

                # Show the frame
                cv2.imshow('Emotion Detection', frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        
        finally:
            self.executor.shutdown(wait=False)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "_main_":
    detector = EmotionDetector()
    detector.run()