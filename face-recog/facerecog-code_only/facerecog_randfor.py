import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt
from scipy import ndimage

def augment_image(image):
    """Apply various augmentations to a face image"""
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Rotation variations
    for angle in [-10, -5, 5, 10]:
        rotated = ndimage.rotate(image, angle, reshape=False)
        augmented_images.append(rotated)
    
    # Brightness variations
    for factor in [0.8, 1.2]:
        brightened = np.clip(image * factor, 0, 255).astype(np.uint8)
        augmented_images.append(brightened)
    
    # Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # Add slight gaussian noise
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    augmented_images.append(noisy)
    
    # Slight zoom in
    h, w = image.shape
    zoom = cv2.resize(image[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)], (w, h))
    augmented_images.append(zoom)
    
    return augmented_images

def main():
    # Path dataset - ADD AYAH HERE
    dataset_paths = {
        'alvin': r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\dataset\alvin',
        'mama': r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\dataset\mama',
        'ayah': r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\dataset\ayah'  # Added new dataset
    }
    
    # Inisialisasi detector wajah dari OpenCV
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Persiapkan data pelatihan
    print("Mempersiapkan data pelatihan...")
    X = []  # gambar wajah
    y = []  # label (alvin atau mama atau ayah)
    
    # Loop melalui setiap dataset
    for person, dataset_path in dataset_paths.items():
        print(f"Memproses dataset: {person}")
        # Pastikan path ada
        if not os.path.exists(dataset_path):
            print(f"Path tidak ditemukan: {dataset_path}")
            continue
            
        # Loop melalui semua file gambar di folder dataset
        image_count = 0
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
                    
                    # Augmentasi gambar
                    augmented_faces = augment_image(face_img)
                    
                    # Tambahkan gambar asli dan hasil augmentasi ke data pelatihan
                    for aug_face in augmented_faces:
                        X.append(aug_face.flatten())  # Flatten menjadi vektor 1D
                        y.append(person)
                    
                    image_count += len(augmented_faces)
                    print(f"Proses gambar: {img_file}, terdeteksi wajah: Ya, (+ {len(augmented_faces)} augmentasi)")
                else:
                    print(f"Proses gambar: {img_file}, terdeteksi wajah: Tidak")
        
        print(f"Total gambar untuk {person} setelah augmentasi: {image_count}")
    
    if not X:
        print("Tidak ada wajah yang terdeteksi dari dataset! Silakan periksa gambar dan folder.")
        return
        
    # Konversi list ke array numpy
    X = np.array(X)
    y = np.array(y)
    
    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Ukuran dataset setelah augmentasi - Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
    
    # Latih model Random Forest
    print("Melatih model Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Gunakan semua core CPU
    )
    model.fit(X_train, y_train)
    
    # Evaluasi model
    print("Evaluasi model...")
    y_pred = model.predict(X_test)
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importances = model.feature_importances_
    print(f"\nTop 10 piksel paling penting:")
    top_features = np.argsort(feature_importances)[-10:]
    for i, idx in enumerate(top_features):
        print(f"Peringkat {i+1}: Piksel {idx}, Importance: {feature_importances[idx]:.6f}")
    
    # Simpan model
    print("Menyimpan model...")
    # Buat direktori untuk model jika belum ada
    model_dir = r'C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'face_recognition_model_rf.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    # Simpan face detector sebagai referensi
    detector_path = os.path.join(model_dir, 'face_detector.pkl')
    with open(detector_path, 'wb') as f:
        pickle.dump({'face_detector': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'}, f)
    
    print(f"Model berhasil disimpan di: {model_path}")
    print(f"Face detector referensi disimpan di: {detector_path}")
    
    # Validasi silang sederhana untuk memeriksa overfitting
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"\nAkurasi training: {train_accuracy:.4f}")
    print(f"Akurasi testing: {test_accuracy:.4f}")
    print(f"Selisih (indikasi overfitting jika besar): {train_accuracy - test_accuracy:.4f}")
    
    # Tampilkan grafik distribusi data
    plt.figure(figsize=(10, 6))
    labels, counts = np.unique(y, return_counts=True)
    plt.bar(labels, counts)
    plt.title('Distribusi Kelas Dataset')
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Sampel')
    plt.savefig(os.path.join(model_dir, 'data_distribution.png'))
    plt.close()
    
    # Simpan informasi model
    model_info = {
        'classes': list(model.classes_),
        'n_features': X.shape[1],
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'image_size': (100, 100)  # Ukuran gambar yang digunakan untuk pelatihan
    }
    
    with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Informasi model disimpan di: {os.path.join(model_dir, 'model_info.pkl')}")

if __name__ == "__main__":
    main()