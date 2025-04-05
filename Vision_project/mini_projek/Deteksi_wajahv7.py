import cv2
import mediapipe as mp
import numpy as np

class FaceHandDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.running = True
        self.confirm_exit = False
        self.option_selected = False
        self.cancel_selected = False
        
        self.selected_color = None
        self.drawing_mode = False
        self.prev_point = None
        self.drawing_layer = None
        
        self.colors = {
            'deep_pink': (147, 20, 255),
            'gold_yellow': (0, 215, 255),
            'pure_black': (0, 0, 0)
        }

    def setup_camera(self):
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return camera

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 10 <= x <= 110 and 10 <= y <= 50:
                self.confirm_exit = True
            
            elif 960 <= x <= 1120 and 10 <= y <= 50:
                self.option_selected = True
            
            elif self.option_selected and 300 <= y <= 400:
                if 300 <= x <= 500:
                    self.selected_color = self.colors['deep_pink']
                    self.drawing_mode = True
                    self.option_selected = False
                elif 500 <= x <= 700:
                    self.selected_color = self.colors['gold_yellow']
                    self.drawing_mode = True
                    self.option_selected = False
                elif 700 <= x <= 900:
                    self.selected_color = self.colors['pure_black']
                    self.drawing_mode = True
                    self.option_selected = False
                elif 900 <= x <= 1100:
                    self.clear_drawing()
                    self.option_selected = False
            
            elif self.confirm_exit and 300 <= x <= 500 and 300 <= y <= 400:
                self.running = False
            
            elif self.confirm_exit and 500 <= x <= 700 and 300 <= y <= 400:
                self.confirm_exit = False

    def init_drawing_layer(self, frame_shape):
        if self.drawing_layer is None:
            self.drawing_layer = np.zeros(frame_shape, dtype=np.uint8)

    def clear_drawing(self):
        if self.drawing_layer is not None:
            self.drawing_layer.fill(0)
        self.selected_color = None
        self.drawing_mode = False
        self.prev_point = None

    def check_color_selection(self, x_tip, y_tip):
        if self.option_selected and 300 <= y_tip <= 400:
            if 300 <= x_tip <= 500:
                self.selected_color = self.colors['deep_pink']
                self.drawing_mode = True
                self.option_selected = False
            elif 500 <= x_tip <= 700:
                self.selected_color = self.colors['gold_yellow']
                self.drawing_mode = True
                self.option_selected = False
            elif 700 <= x_tip <= 900:
                self.selected_color = self.colors['pure_black']
                self.drawing_mode = True
                self.option_selected = False

    def check_hand_gestures(self, landmarks, frame_shape):
        frame_width, frame_height = frame_shape[1], frame_shape[0]
        
        index_tip = landmarks[8]
        x_tip, y_tip = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
        
        pinky_tip = landmarks[20]
        x_pinky, y_pinky = int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)

        self.init_drawing_layer(frame_shape)
        
        if x_tip < 110 and 10 <= y_tip <= 50:
            self.confirm_exit = True
        
        if self.confirm_exit:
            if 300 <= x_tip <= 500 and 300 <= y_tip <= 400:
                self.running = False
            elif 500 <= x_tip <= 700 and 300 <= y_tip <= 400:
                self.confirm_exit = False
        
        if 960 <= x_pinky <= 1120 and 10 <= y_pinky <= 50:
            self.option_selected = True
            
        if (self.option_selected or self.drawing_mode) and \
           900 <= x_tip <= 1100 and 300 <= y_tip <= 400:
            self.option_selected = False
            self.clear_drawing()

        self.check_color_selection(x_tip, y_tip)

        if self.drawing_mode and self.selected_color is not None:
            if self.prev_point is None:
                self.prev_point = (x_tip, y_tip)
            else:
                cv2.line(self.drawing_layer, self.prev_point, (x_tip, y_tip), 
                        self.selected_color, thickness=2)
                self.prev_point = (x_tip, y_tip)

    def draw_ui(self, frame, num_faces):
        cv2.rectangle(frame, (10, 10), (110, 50), (0, 0, 255), -1)
        cv2.putText(frame, "EXIT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (960, 10), (1120, 50), (0, 255, 255), -1)
        cv2.putText(frame, "Option", (970, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        text = f"Terdeteksi = {num_faces}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 20
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if self.option_selected:
            cv2.rectangle(frame, (300, 300), (500, 400), self.colors['deep_pink'], -1)
            cv2.rectangle(frame, (500, 300), (700, 400), self.colors['gold_yellow'], -1)
            cv2.rectangle(frame, (700, 300), (900, 400), self.colors['pure_black'], -1)
            
            cv2.rectangle(frame, (900, 300), (1100, 400), (0, 0, 255), -1)
            cv2.putText(frame, "Cancel", (970, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.confirm_exit:
            cv2.rectangle(frame, (300, 300), (500, 400), (0, 255, 0), -1)
            cv2.putText(frame, "Yes", (370, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (500, 300), (700, 400), (0, 0, 255), -1)
            cv2.putText(frame, "No", (550, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def run(self):
        camera = self.setup_camera()
        cv2.namedWindow('Deteksi Wajah dan Jari', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Deteksi Wajah dan Jari', self.mouse_click)

        while self.running:
            ret, frame = camera.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, 
                                                     minNeighbors=5, minSize=(150, 150))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    self.check_hand_gestures(landmarks.landmark, frame.shape)
            else:
                self.prev_point = None
            
            if self.drawing_layer is not None:
                frame = cv2.addWeighted(frame, 1.0, self.drawing_layer, 1.0, 0)
            
            self.draw_ui(frame, len(faces))
            
            cv2.imshow('Deteksi Wajah dan Jari', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceHandDetector()
    detector.run()