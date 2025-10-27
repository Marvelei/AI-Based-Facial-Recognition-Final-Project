# recognition_app.py
# FINAL VERSION: Advanced DeepFace Recognition System - Fully Automated Batch Processor

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

# Note: img_a and img_b now default to None and are used ONLY for the Verification section.
parser.add_argument("--img_a", default=None, help="Path to test image A (Optional for 1:1 Verification)")
parser.add_argument("--img_b", default=None, help="Path to test image B (Optional for 1:1 Verification)")
parser.add_argument("--db", default="facial_db", help="Path to face database for Recognition")
parser.add_argument("--model", default="ArcFace", help="Model to use (VGG-Face, Facenet, ArcFace, etc.)")
parser.add_argument("--detector", default="retinaface", help="Face detector (opencv, mtcnn, mediapipe, retinaface)")
parser.add_argument("--metric", default="euclidean_l2", help="Distance metric (cosine, euclidean, euclidean_l2)")
parser.add_argument("--show", action="store_true", help="Show face extraction window for the first image processed")

args = parser.parse_args()

# --- VALIDATION CHECK (Only validating the mandatory database path) ---
def check_file_exists(path, label):
    if not os.path.exists(path):
        print(f"\nFATAL ERROR: The required file/folder path for {label} does not exist!")
        print(f"Please confirm the following path is correct and the file/folder is present:")
        print(f"-> {os.path.abspath(path)}")
        sys.exit(1)
    return True

# ONLY check the database path, as the images are optional inputs for the loop
check_file_exists(args.db, "Database") 


# ---------------------------
# 2. MODEL BUILDING (Logging)
# ---------------------------
print(f"🚀 Using DeepFace with model: {args.model}")
print(f"🚀 Using face detector: {args.detector}")


# ---------------------------
# 3. FACE VERIFICATION (Conditional Check)
# ---------------------------
print(f"\n--- 1. FACE VERIFICATION (1:1) ---")
if args.img_a and args.img_b:
    try:
        # Check files exist before running verification
        check_file_exists(args.img_a, "Image A for Verification")
        check_file_exists(args.img_b, "Image B for Verification")
        
        result_verify = DeepFace.verify(
            img1_path=args.img_a,
            img2_path=args.img_b,
            model_name=args.model,
            distance_metric=args.metric,
            detector_backend=args.detector
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
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return json.JSONEncoder.default(self, obj) # Fallback to default encoder for other types

# Iterate over all files in the test folder
for filename in os.listdir(TEST_FOLDER_PATH):
    # Only process common image files
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        
        image_path = os.path.join(TEST_FOLDER_PATH, filename)
        
        # Skip processing if file is corrupted/unreadable (prevents crash)
        if not os.path.isfile(image_path):
             print(f"  ⚠️ Skipping {filename}: Not a valid file.")
             continue
             
        processed_count += 1
        print(f"\n>>> PROCESSING IMAGE {processed_count}: {image_path} <<<")
        
        # --- 4.1. FACE RECOGNITION (Find Identity) ---
        try:
            dfs = DeepFace.find(
                img_path=image_path,
                db_path=args.db,
                model_name=args.model,
                detector_backend=args.detector
            )

            if dfs and isinstance(dfs, list) and not dfs[0].empty:
                match_identity = dfs[0]['identity'].iloc[0].split(os.sep)[-2]
                print(f"  🎉 MATCH FOUND: {match_identity}")
                print(dfs[0][['identity', 'distance']].head(1).to_string(index=False)) 
            else:
                print("  🤷‍♂️ No match found in database.")
        except Exception as e:
            print(f"  ❌ Recognition Error: {e}")

        # --- 4.2. FACIAL ATTRIBUTE ANALYSIS ---
        try:
            analysis_results = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'gender', 'emotion', 'race'],
                detector_backend=args.detector
            )

            result = analysis_results[0]
            print(f"  Age: {result['age']}, Gender: {result['dominant_gender']}")
            print(f"  Emotion: {result['dominant_emotion']}, Race: {result['dominant_race']}")

            # Save results 
            os.makedirs("outputs/analysis", exist_ok=True)
            output_filename = os.path.splitext(filename)[0] + "_analysis.json"
            with open(os.path.join("outputs/analysis", output_filename), "w") as f:
                json.dump(result, f, indent=4, default=numpy_encoder)
            
        except Exception as e:
            print(f"  ❌ Analysis Error: {e}")

        # --- 4.3. FACE EMBEDDING (FIXED tolist ERROR) ---
        try:
            embeddings = DeepFace.represent(
                img_path=image_path,
                model_name=args.model,
                detector_backend=args.detector,
            )
            
            # Extract the embedding list/array directly
            embedding_vector = embeddings[0]['embedding']
            
            # Prepare serializable data
            serializable_data = embeddings[0].copy()
            
            # **SAFETY CHECK:** Convert to list ONLY IF it is a NumPy array (np.ndarray)
            if isinstance(embedding_vector, np.ndarray):
                serializable_data['embedding'] = embedding_vector.tolist()
            else:
                # If it's already a list (which DeepFace sometimes returns), assign it directly
                serializable_data['embedding'] = embedding_vector

            print(f"  Embedding vector length: {len(serializable_data['embedding'])}")

            # Save a serializable copy
            os.makedirs("outputs/embeddings", exist_ok=True)
            output_filename = os.path.splitext(filename)[0] + "_embedding.json"
            with open(os.path.join("outputs/embeddings", output_filename), "w") as f:
                json.dump(serializable_data, f, indent=4)
            print(f"  📁 Saved embedding to outputs/embeddings/{output_filename}")

        except Exception as e:
            print(f"  ❌ Embedding Error: {e}")

# ---------------------------
# 5. FACE EXTRACTION & VISUALIZATION (Runs ONLY on the first image processed, or args.img_a)
# ---------------------------
print(f"\n--- 3. FACE EXTRACTION & VISUALIZATION ---")
# This section now runs on the first image in the test_images folder, 
# or the image specified by --img_a, if provided.
img_to_show = args.img_a if args.img_a else os.path.join(TEST_FOLDER_PATH, os.listdir(TEST_FOLDER_PATH)[0])

if args.show and os.path.isfile(img_to_show):
    try:
        faces = DeepFace.extract_faces(
            img_path=img_to_show,
            detector_backend=args.detector,
            align=True,
        )

        print(f"Detected {len(faces)} face(s) in {img_to_show}.")

        img = cv2.imread(os.path.abspath(img_to_show))
        
        if img is None:
            print(f"Visualization Error: Could not load image for drawing. Check file integrity.")
        else:
            for f in faces:
                area = f['facial_area']
                x, y, w, h = area['x'], area['y'], area['w'], area['h']
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow(f"Detected Faces in {os.path.basename(img_to_show)} (Press 'q' to close)", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Extraction/Visualization Error: {e}")
else:
    print("ℹ️ Visualization skipped: Pass --show argument to enable the display.")

print("\n✅ All tasks completed successfully!")