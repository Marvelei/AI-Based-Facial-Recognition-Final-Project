import insightface
import cv2
import time

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)   # <--- GPU MODE

img = cv2.imread("test.jpg")
t = time.time()
faces = model.get(img)
print("Time:", time.time() - t)
