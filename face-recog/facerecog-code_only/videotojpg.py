import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        if frame_num % frame_skip == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Disimpan: {filename}")
            saved_count += 1

        frame_num += 1

    cap.release()
    print(f"Total frame disimpan: {saved_count}")


video_path = r"C:\Users\alvin\Documents\vscode_apin\python\comvis\mamadataB.mp4"  # Ganti dengan path video kamu
output_folder = r"C:\Users\alvin\Documents\vscode_apin\python\comvis\face-recog\dataset\mama"  # Ganti dengan folder tujuan
extract_frames(video_path, output_folder, frame_skip=1)  # Ambil tiap 2 frame

#penjelasan frame_skip (jumlah gambar yang diperoleh)
#frame_skip=1 -> 30 (gambar tiap detik )/ 1 (frame_skip)