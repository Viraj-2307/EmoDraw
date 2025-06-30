import cv2
import mediapipe as mp
import numpy as np
import torch
import open_clip
from PIL import Image
import torch.nn.functional as F

# Load model and transforms
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
model.eval()

# Define labels you want to predict
labels = ["cat", "dog", "tree", "star", "house", "sun", "car", "person", "bottle", "airplane", "cup", "phone","triangle", "square", "circle", "rectangle", "pentagon", "hexagon", "octagon"]

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize variables
mode = 'idle'  # Can be 'draw', 'erase', 'idle'
drawn_points = []
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

def predict_with_openclip(image_path, labels):
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    text_inputs = tokenizer(labels).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)
    
    best_idx = probs.argmax().item()
    return labels[best_idx], probs[0, best_idx].item()

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
                        if distance > 20:  # Eraser size (20 px radius)
                            new_points.append(point)
                    drawn_points = new_points

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
            label, confidence = predict_with_openclip('drawing.png', labels)
            print(f'Predicted: {label} ({confidence * 100:.2f}%)')
            cv2.putText(combined, f'{label} ({confidence * 100:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Prediction', combined)
            cv2.waitKey(2000)
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
