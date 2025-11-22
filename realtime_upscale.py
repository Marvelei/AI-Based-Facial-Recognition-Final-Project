import cv2
import numpy as np
import sys
import os
import torch # Now a primary focus for GPU acceleration
import warnings
from deepface import DeepFace
from gfpgan import GFPGANer

# Suppress PyTorch/TensorFlow warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
DETECTOR = "retinaface" 
CAMERA_INDEX = 0 
OUTPUT_SCALE = 2 # 2x Upscale

# --- 2. INITIALIZE MODELS ---
# Set the TensorFlow/DeepFace device explicitly to CPU to prevent failure on GPU memory allocation
# Note: DeepFace is not performance bottlenecked here, so running it on CPU is acceptable.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

try:
    # DeepFace Initialization (TensorFlow/CPU)
    print("🚀 Initializing DeepFace Detector (RetinaFace) on CPU...")
    _ = DeepFace.extract_faces(
        img_path=np.zeros((100, 100, 3), dtype=np.uint8),
        detector_backend=DETECTOR,
        enforce_detection=False
    )
    print("✅ DeepFace Detector Ready.")
except Exception as e:
    print(f"FATAL: DeepFace (TF) failed to initialize: {e}")
    sys.exit(1)


try:
    # GFPGAN Initialization (PyTorch/GPU)
    # Check if GPU is available in the PyTorch environment
    gfpgan_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if gfpgan_device == 'cuda':
        print(f"🤖 Initializing GFPGAN (Device: {gfpgan_device} - Using RTX 4060)")
        
        # --- CRUCIAL GPU CHECK ---
        # Verify the PyTorch version is correct for your CUDA setup (usually 11.8 for current stable)
        if torch.cuda.get_device_properties(0).major < 8: # RTX 4060 is Compute Capability 8.6
            print("⚠️ Warning: PyTorch version may not fully support Ada Lovelace architecture.")
        
    else:
        # If CUDA is not available, proceed on CPU but notify user
        print(f"🤖 Initializing GFPGAN (Device: {gfpgan_device} - Warning: CPU only)")

    # FIX: Use the stable model URL (v1.3) and the correct architecture name
    restorer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', 
        upscale=OUTPUT_SCALE,
        arch='GFPGANv1', 
        channel_multiplier=2,
        bg_upsampler=None, 
        device=gfpgan_device
    )
    print(f"✅ GFPGAN Restorer Ready.")
except Exception as e:
    print(f"FATAL: GFPGAN failed to initialize: {e}")
    sys.exit(1)


# --- 3. VIDEO STREAM LOOP ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"FATAL: Cannot open camera with index {CAMERA_INDEX}. Please ensure the camera is not in use by another application.")
    exit()

print("\n--- STARTING REAL-TIME FACE UPSCALE (Press 'q' to quit) ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    restored_face = None
    
    try:
        # 3.1. DETECT AND CROP FACE (This is fast enough on CPU)
        detected_faces = DeepFace.extract_faces(
            img_path=frame, 
            detector_backend=DETECTOR, 
            enforce_detection=False
        )

        if detected_faces:
            face_data = detected_faces[0]
            area = face_data['facial_area']
            
            p = 10 
            x, y, w, h = area['x']-p, area['y']-p, area['w']+2*p, area['h']+2*p
            
            x, y = max(0, x), max(0, y)
            w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)

            cropped_face = frame[y:y+h, x:x+w]

            # 3.2. RESTORE AND UPSCALE (This is the heavy lifting done on the RTX 4060)
            _, _, restored_output = restorer.enhance(
                cropped_face, 
                has_aligned=False, 
                only_center_face=True, 
                paste_back=False
            )
            
            if restored_output and len(restored_output) > 0:
                restored_face = restored_output[0]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    except Exception as e:
        pass

    # 3.3. DISPLAY RESULTS
    cv2.imshow('Original Frame (DeepFace Detection)', frame)
    
    if restored_face is not None:
        cv2.imshow(f'GFPGAN Enhanced Face ({OUTPUT_SCALE}x)', restored_face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Real-time stream stopped.")