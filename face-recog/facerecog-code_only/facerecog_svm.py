import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt

def main():
    # Path dataset
    dataset_paths = {
        'alvin': r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\dataset\alvin',
        'mama': r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\dataset\mama'
    }
    
    # Inisialisasi detector wajah dari OpenCV
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Persiapkan data pelatihan
    print("Mempersiapkan data pelatihan...")
    X = []  # gambar wajah
    y = []  # label (alvin atau mama)
    
    # Loop melalui setiap dataset
    for person, dataset_path in dataset_paths.items():
        print(f"Memproses dataset: {person}")
        # Pastikan path ada
        if not os.path.exists(dataset_path):
            print(f"Path tidak ditemukan: {dataset_path}")
            continue
            
        # Loop melalui semua file gambar di folder dataset
        for img_file in os.listdir(dataset_path):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dataset_path, img_file)
                
                # Baca gambar
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Tidak dapat membaca gambar: {img_path}")
                    continue
                
                # Konversi ke grayscale untuk deteksi wajah
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Deteksi wajah
                faces = face_detector.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) > 0:
                    # Ambil wajah pertama yang terdeteksi
                    x, y_coord, w, h = faces[0]
                    
                    # Crop wajah
                    face_img = gray[y_coord:y_coord+h, x:x+w]
                    
                    # Resize ke ukuran yang sama (100x100 piksel)
                    face_img = cv2.resize(face_img, (100, 100))
                    
                    # Tambahkan ke data pelatihan
                    X.append(face_img.flatten())  # Flatten menjadi vektor 1D
                    y.append(person)
                    
                    print(f"Proses gambar: {img_file}, terdeteksi wajah: Ya")
                else:
                    print(f"Proses gambar: {img_file}, terdeteksi wajah: Tidak")
    
    if not X:
        print("Tidak ada wajah yang terdeteksi dari dataset! Silakan periksa gambar dan folder.")
        return
        
    # Konversi list ke array numpy
    X = np.array(X)
    y = np.array(y)
    
    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Latih model SVM (Support Vector Machine)
    print("Melatih model...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Evaluasi model
    print("Evaluasi model...")
    y_pred = model.predict(X_test)
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    
    # Simpan model
    print("Menyimpan model...")
    model_path = r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\face_recognition_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    # Simpan face detector sebagai referensi
    detector_path = r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\face_detector.pkl'
    with open(detector_path, 'wb') as f:
        pickle.dump({'face_detector': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'}, f)
    
    print(f"Model berhasil disimpan di: {model_path}")
    print(f"Face detector referensi disimpan di: {detector_path}")

# Fungsi untuk pengujian dengan gambar baru
def test_model_with_image(model_path, img_path):
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load face detector
    with open(r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\face_detector.pkl', 'rb') as f:
        detector_data = pickle.load(f)
    
    face_detector = cv2.CascadeClassifier(detector_data['face_detector'])
    
    # Baca gambar
    img = cv2.imread(img_path)
    if img is None:
        print(f"Tidak dapat membaca gambar: {img_path}")
        return
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = face_detector.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        # Crop dan proses wajah
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        
        # Prediksi
        face_encoding = face_img.flatten().reshape(1, -1)
        prediction = model.predict(face_encoding)[0]
        probability = np.max(model.predict_proba(face_encoding))
        
        # Tampilkan hasil
        label = f"{prediction} ({probability:.2f})"
        color = (0, 255, 0) if probability > 0.7 else (0, 165, 255)
        
        # Gambar kotak dan label
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Tampilkan hasil
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Hasil Deteksi')
    plt.show()

if __name__ == "__main__":
    main()