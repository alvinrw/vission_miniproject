import cv2
import numpy as np
import pickle

# Load model yang sudah dilatih
model_path = r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\face_recognition_model.pkl'
detector_path = r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\face_detector.pkl'

# Load model
print("Memuat model...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load face detector
with open(detector_path, 'rb') as f:
    detector_data = pickle.load(f)

face_detector = cv2.CascadeClassifier(detector_data['face_detector'])

def recognize_face(frame):
    # Buat salinan frame untuk output
    output_frame = frame.copy()
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop untuk setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Crop dan proses wajah
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        
        # Flatten gambar untuk prediksi
        face_encoding = face_img.flatten().reshape(1, -1)
        
        # Prediksi
        prediction = model.predict(face_encoding)[0]
        probability = np.max(model.predict_proba(face_encoding))
        
        # Tentukan label dan warna berdasarkan prediksi
        if probability > 0.6:  # Ambang batas kepercayaan
            # Jika dikenali sebagai alvin atau mama
            label = prediction
            color = (0, 255, 0)  # Hijau
        else:
            # Jika tidak dikenali
            label = "Wajah tak dikenali"
            color = (0, 255, 255)  # Kuning
        
        # Tambahkan nilai probabilitas jika dikenali
        if label in ["alvin", "mama"]:
            label = f"{label} ({probability:.2f})"
        
        # Gambar kotak dan label
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
        
        # Siapkan background untuk teks
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(output_frame, (x, y-text_size[1]-10), (x+text_size[0], y), color, -1)
        
        # Tambahkan teks
        cv2.putText(output_frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return output_frame

# Fungsi untuk menggunakan webcam
def webcam_recognition():
    print("Memulai pengenalan wajah dengan webcam...")
    print("Tekan 'q' untuk keluar.")
    
    # Buka webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame dari webcam.")
            break
        
        # Kenali wajah dalam frame
        output_frame = recognize_face(frame)
        
        # Tampilkan hasil
        cv2.imshow('Pengenalan Wajah', output_frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Bersihkan
    cap.release()
    cv2.destroyAllWindows()

# Fungsi untuk pengenalan pada gambar
def image_recognition(image_path):
    print(f"Mengenali wajah dalam gambar: {image_path}")
    
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Tidak dapat membaca gambar: {image_path}")
        return
    
    # Kenali wajah dalam gambar
    output_image = recognize_face(image)
    
    # Tampilkan hasil
    cv2.imshow('Hasil Pengenalan', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Pilih mode operasi
    mode = input("Pilih mode (1: Webcam, 2: Gambar): ")
    
    if mode == "1":
        webcam_recognition()
    elif mode == "2":
        image_path = input("Masukkan path gambar: ")
        image_recognition(image_path)
    else:
        print("Mode tidak valid. Silakan pilih 1 atau 2.")