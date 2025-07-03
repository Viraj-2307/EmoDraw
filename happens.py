import cv2
import mediapipe as mp
import numpy as np
import torch
import open_clip
import faiss
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)
model.eval()

# Load FAISS embeddings and labels
clip_embeddings = np.load('clip_embeddings.npy').astype('float32')
clip_labels = np.load('clip_labels.npy')

# Build FAISS index
index = faiss.IndexFlatIP(clip_embeddings.shape[1])  # Cosine similarity with normalized vectors

# Normalize embeddings for cosine search
clip_embeddings /= np.linalg.norm(clip_embeddings, axis=1, keepdims=True)
index.add(clip_embeddings)

print(f"✅ FAISS index built with {clip_embeddings.shape[0]} samples.")

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

mode = 'idle'
drawn_points = []
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

def get_image_embedding(image_path):
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype('float32')

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                if mode == 'draw':
                    drawn_points.append((cx, cy))

                elif mode == 'erase':
                    new_points = []
                    for point in drawn_points:
                        px, py = point
                        distance = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                        if distance > 20:
                            new_points.append(point)
                    drawn_points = new_points

        # Draw points
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        for point in drawn_points:
            cv2.circle(canvas, point, 5, (255, 0, 0), -1)

        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        cv2.imshow('EmoDraw - D: Draw | E: Erase | C: Clear | P: Predict | Q: Quit', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            mode = 'draw'
        elif key == ord('e'):
            mode = 'erase'
        elif key == ord('c'):
            drawn_points = []
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        elif key == ord('p'):
            cv2.imwrite('drawing.png', canvas)
            query_embedding = get_image_embedding('drawing.png')

            # Search using FAISS
            D, I = index.search(query_embedding, 1)  # top-1 result

            best_match = clip_labels[I[0][0]]
            confidence = D[0][0]

            if confidence > 0.3:  # Set a reasonable threshold
                result_text = f'{best_match} ({confidence * 100:.2f}%)'
            else:
                result_text = 'Not Recognized'

            print(f'🧠 Prediction: {result_text}')
            cv2.putText(combined, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Prediction', combined)
            cv2.waitKey(2000)
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
