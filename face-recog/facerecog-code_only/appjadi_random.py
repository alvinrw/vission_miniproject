import os
import cv2
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def load_model():
    """Load the trained face recognition model and associated info"""
    model_dir = r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\models'
    
    # Load the face recognition model
    model_path = os.path.join(model_dir, 'face_recognition_model_rf.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load model info
    model_info_path = os.path.join(model_dir, 'model_info.pkl')
    with open(model_info_path, 'rb') as f:
        model_info = pickle.load(f)
    
    # Load face detector reference
    detector_path = os.path.join(model_dir, 'face_detector.pkl')
    with open(detector_path, 'rb') as f:
        detector_info = pickle.load(f)
    
    # Load the face detector
    face_detector = cv2.CascadeClassifier(detector_info['face_detector'])
    
    return model, model_info, face_detector

def process_image(image, model, model_info, face_detector, confidence_threshold=0.6):
    """Process a single image and return the image with face recognition results"""
    # Convert to grayscale for face detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.1, 5)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face_img = gray[y:y+h, x:x+w]
        
        # Skip if face is too small
        if face_img.shape[0] < 30 or face_img.shape[1] < 30:
            continue
            
        face_img = cv2.resize(face_img, model_info['image_size'])
        face_vector = face_img.flatten().reshape(1, -1)
        
        # Predict identity
        identity = model.predict(face_vector)[0]
        confidence = np.max(model.predict_proba(face_vector)[0])
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            identity = "Data tak dikenali"
        
        # Draw rectangle and label
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Add text with prediction
        cv2.putText(image, identity, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image

def webcam_recognition(model, model_info, face_detector):
    """Run face recognition using webcam"""
    print("Opening webcam for face recognition...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    print("Press 'q' to quit")
    
    # Set minimum confidence threshold
    confidence_threshold = 0.6
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process the frame
        processed_frame = process_image(frame, model, model_info, face_detector, confidence_threshold)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition', processed_frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def upload_image_recognition(model, model_info, face_detector):
    """Run face recognition from manual path input"""
    import os

    while True:
        file_path = input("Masukkan path gambar (atau ketik 'selesai' untuk keluar): ").strip()
        if file_path.lower() == 'selesai':
            break

        if not os.path.isfile(file_path):
            print("❌ File tidak ditemukan. Coba lagi.")
            continue

        image = cv2.imread(file_path)
        if image is None:
            print("⚠️ Gagal membaca gambar.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            print(f"Gambar {os.path.basename(file_path)}: ❌ Tidak ada wajah terdeteksi.")
            continue

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            if face_img.shape[0] < 30 or face_img.shape[1] < 30:
                print(f"Gambar {os.path.basename(file_path)}: ⚠️ Wajah terlalu kecil.")
                continue

            face_img = cv2.resize(face_img, model_info['image_size'])
            face_vector = face_img.flatten().reshape(1, -1)

            identity = model.predict(face_vector)[0]
            confidence = np.max(model.predict_proba(face_vector)[0])

            if confidence < 0.5:
                identity = "Wajah tak dikenali"

            print(f"Gambar {os.path.basename(file_path)}: Ini {identity}")


def show_menu():
    """Display menu and get user choice"""
    print("\nFace Recognition System")
    print("1. Use Webcam")
    print("2. Upload Images")
    print("3. Exit")
    
    while True:
        choice = input("Enter your choice (1-3): ")
        if choice in ['1', '2', '3']:
            return int(choice)
        print("Invalid choice. Please try again.")

def main():
    print("Loading face recognition model...")
    try:
        model, model_info, face_detector = load_model()
        print("Model loaded successfully!")
        print(f"Classes: {model_info['classes']}")
        
        while True:
            choice = show_menu()
            
            if choice == 1:
                webcam_recognition(model, model_info, face_detector)
            elif choice == 2:
                upload_image_recognition(model, model_info, face_detector)
            else:
                print("Exiting program.")
                break
                
    except Exception as e:
        print(f"Error: {e}")
        print("Check if the model files exist at the specified path.")

if __name__ == "__main__":
    main()