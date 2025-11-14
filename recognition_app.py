# recognition_app.py
# FINAL VERSION: Advanced DeepFace Recognition System - Fully Automated & Stable

import cv2
import json
import argparse
from deepface import DeepFace
import os
import sys
import numpy as np 

# ---------------------------
# 1. ARGUMENT PARSER
# ---------------------------
parser = argparse.ArgumentParser(description="Advanced DeepFace Recognition System")

parser.add_argument("--img_a", default=None, help="Path to test image A (Optional for 1:1 Verification)")
parser.add_argument("--img_b", default=None, help="Path to test image B (Optional for 1:1 Verification)")
parser.add_argument("--db", default="facial_db", help="Path to face database for Recognition")
parser.add_argument("--model", default="ArcFace", help="Model to use (VGG-Face, Facenet, ArcFace, etc.)")
# Detector set to stable 'retinaface' for high accuracy and stability
parser.add_argument("--detector", default="retinaface", help="Face detector (opencv, mtcnn, mediapipe, retinaface)")
parser.add_argument("--metric", default="euclidean_l2", help="Distance metric (cosine, euclidean, euclidean_l2)")
parser.add_argument("--show", action="store_true", help="Show face extraction window for the first image processed")

args = parser.parse_args()

# --- NEW: DEFINE ALL DATABASE PATHS ---
DB_PATHS = ["baseline_db", "masked_db"] # Use these new paths

# --- VALIDATION CHECK (Only validating the mandatory database path) ---
def check_file_exists(path, label):
    if not os.path.exists(path):
        print(f"\nFATAL ERROR: The required file/folder path for {label} does not exist!")
        print(f"Please confirm the following path is correct and the file/folder is present:")
        print(f"-> {os.path.abspath(path)}")
        sys.exit(1)
    return True

# Validate the NEW database paths
check_file_exists(DB_PATHS[0], "Baseline Database")
check_file_exists(DB_PATHS[1], "Masked Database")


# ---------------------------
# 2. CORE LOGGING
# ---------------------------
print(f"🚀 Using DeepFace with model: {args.model}")
print(f"🚀 Using face detector: {args.detector}")


# ---------------------------
# 3. FACE VERIFICATION (Conditional Check)
# ---------------------------
print(f"\n--- 1. FACE VERIFICATION (1:1) ---")
if args.img_a and args.img_b:
    try:
        check_file_exists(args.img_a, "Image A for Verification")
        check_file_exists(args.img_b, "Image B for Verification")
        
        result_verify = DeepFace.verify(
            img1_path=args.img_a,
            img2_path=args.img_b,
            model_name=args.model,
            distance_metric=args.metric,
            detector_backend=args.detector,
            anti_spoofing=True, 
            enforce_detection=False 
        )

        verified = result_verify['verified']
        print("✅ SAME PERSON" if verified else "❌ DIFFERENT PEOPLE")
        print(f"Distance: {result_verify['distance']:.4f} | Threshold: {result_verify['threshold']}")
    except Exception as e:
        print(f"Verification Error: {e}")
else:
    print("ℹ️ Verification skipped: Use --img_a and --img_b arguments to run 1:1 match.")

# =======================================================
# 4. BATCH PROCESSING LOOP
# =======================================================

TEST_FOLDER_PATH = "test_images"
processed_count = 0

print(f"\n==================================================")
print(f"--- 2. BATCH RECOGNITION & ANALYSIS OF FOLDER ---")
print(f"==================================================")

# Define a custom JSON encoder function to handle ALL NumPy types
def numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Initialize outside the loop
cv2.namedWindow('Detected Faces') 

# Iterate over all files in the test folder
for filename in os.listdir(TEST_FOLDER_PATH):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        
        image_path = os.path.join(TEST_FOLDER_PATH, filename)
        
        if not os.path.isfile(image_path):
             print(f"  ⚠️ Skipping {filename}: Not a valid file.")
             continue
             
        processed_count += 1
        print(f"\n>>> PROCESSING IMAGE {processed_count}: {image_path} <<<")
        
        # --- 4.1. FACE EXTRACTION (Required for both Viz and Processing) ---
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=args.detector,
                align=True,
                anti_spoofing=True,
                enforce_detection=False
            )
            print(f"  Detected {len(faces)} face(s).")
            
        except Exception as e:
            print(f"  ❌ Extraction Error: {e}")
            faces = [] # Ensure faces is defined as an empty list on failure

        # --- VISUALIZATION (Runs inside the loop if --show is provided) ---
        if args.show and faces:
            try:
                img = cv2.imread(os.path.abspath(image_path))
                
                if img is None:
                    print(f"  ⚠️ Visualization Error: Could not load image for drawing.")
                else:
                    for i, f in enumerate(faces):
                        area = f['facial_area']
                        x, y, w, h = area['x'], area['y'], area['w'], area['h']
                        
                        # Draw green rectangle (0, 255, 0)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Draw Face Number Label
                        label = f"Face {i+1}"
                        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    cv2.setWindowTitle('Detected Faces', f'Batch Image {processed_count}: {filename} (Press Q to quit batch)')
                    cv2.imshow('Detected Faces', img)
                    
                    key = cv2.waitKey(0) & 0xFF 
                    cv2.destroyAllWindows()
                    
                    if key == ord('q'): # Check if the key pressed was 'q'
                        print("\n🛑 Visualization stopped by user. Aborting batch process.")
                        sys.exit(0) # Exit the entire script

            except Exception as e:
                print(f"  ❌ Visualization/OpenCV Error: {e}")
                
        
        # --- 4.2. FACE RECOGNITION (Find Identity) ---
        try:
            recognition_results = []
            
            for db_path in DB_PATHS:
                # Run find function against ONE database path at a time
                dfs_single = DeepFace.find(
                    img_path=image_path,
                    db_path=db_path, 
                    model_name=args.model,
                    detector_backend=args.detector,
                    anti_spoofing=True,         
                    enforce_detection=False     
                )
                if dfs_single and isinstance(dfs_single, list):
                    recognition_results.extend(dfs_single)

            # Process the combined results
            if recognition_results and not recognition_results[0].empty:
                # The count of faces analyzed is len(faces) from the extraction step
                true_face_count = len(faces)
                print(f"  🔍 Detected {true_face_count} unique face(s). Logging {len(recognition_results)} search results.")
                
                # Report results corresponding to the indexed faces
                for i in range(true_face_count):
                    # Find results specific to the i-th face from both searches
                    
                    # We assume DeepFace returns detection results ordered by face position
                    if i < len(recognition_results): 
                        df_result = recognition_results[i]
                        
                        if not df_result.empty:
                            match_identity = df_result['identity'].iloc[0].split(os.sep)[-2]
                            distance = df_result['distance'].iloc[0]
                            
                            db_source = 'Masked' if 'masked_db' in df_result['identity'].iloc[0] else 'Baseline'
                            
                            print(f"    - Face {i+1} Match: **{match_identity}** (Source: {db_source}, Distance: {distance:.4f})")
                        else:
                            print(f"    - Face {i+1} Match: **No match found**.")
                    else:
                        print(f"    - Face {i+1} Match: **Skipped (Detection Index Error)**.") # Safety check
            else:
                print("  🤷‍♂️ No face detected for recognition or the result set was empty.")
            
        except Exception as e:
            print(f"  ❌ Recognition Error: {e}")

        # --- 4.3. FACIAL ATTRIBUTE ANALYSIS ---
        try:
            analysis_results = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'emotion', 'race'], 
                detector_backend=args.detector,
                anti_spoofing=True,         
                enforce_detection=False     
            )

            if analysis_results and isinstance(analysis_results, list):
                print(f"  🔬 Found {len(analysis_results)} face(s) for analysis.")
                
                for i, result in enumerate(analysis_results):
                    print(f"    - Face {i+1} Analysis:")
                    print(f"      Age: {result['age']}, Gender: // Skipped")
                    print(f"      Emotion: {result['dominant_emotion']}, Race: {result['dominant_race']}")
                    
                    os.makedirs("outputs/analysis", exist_ok=True)
                    base_filename = os.path.splitext(filename)[0]
                    output_filename = f"{base_filename}_face{i+1}_analysis.json"
                    
                    serializable_result = result.copy()
                    
                    # Remove gender data before JSON serialization
                    if 'gender' in serializable_result:
                         del serializable_result['gender']
                    if 'dominant_gender' in serializable_result:
                         del serializable_result['dominant_gender']

                    for key, value in serializable_result.items():
                        if isinstance(value, np.ndarray):
                            serializable_result[key] = value.tolist()

                    with open(os.path.join("outputs/analysis", output_filename), "w") as f:
                        json.dump(serializable_result, f, indent=4, default=numpy_encoder) 
            
        except Exception as e:
            print(f"  ❌ Analysis Error: {e}")

        # --- 4.4. FACE EMBEDDING (Vector Generation) ---
        try:
            embeddings = DeepFace.represent(
                img_path=image_path,
                model_name=args.model,
                detector_backend=args.detector,
                anti_spoofing=True,         
                enforce_detection=False     
            )
            
            if embeddings:
                # Save the first embedding (index 0)
                embedding_vector = embeddings[0]['embedding']
                
                serializable_data = embeddings[0].copy()
                
                if isinstance(embedding_vector, np.ndarray):
                    serializable_data['embedding'] = embedding_vector.tolist()
                else:
                    serializable_data['embedding'] = embedding_vector

                print(f"  Embedding vector length (First Face): {len(serializable_data['embedding'])}")

                os.makedirs("outputs/embeddings", exist_ok=True)
                output_filename = os.path.splitext(filename)[0] + "_embedding_face1.json"
                with open(os.path.join("outputs/embeddings", output_filename), "w") as f:
                    json.dump(serializable_data, f, indent=4)
                print(f"  📁 Saved embedding for Face 1 to outputs/embeddings/{output_filename}")
            else:
                 print("  ⚠️ No face found to generate embedding.")

        except Exception as e:
            print(f"  ❌ Embedding Error: {e}")


# ---------------------------
# 5. FINAL CLEANUP
# ---------------------------
cv2.destroyAllWindows() 
print("\n✅ All tasks completed successfully!")