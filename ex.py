# STEP1 : Install Packages
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP2 : Create Task Processor
# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app = FaceAnalysis(alloed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# STEP3 : Get Image : Pre Processing
# img = ins_get_image('t1')
# img =cv2.imread('imgs/twice.jpg')
img1 =cv2.imread('imgs/IU1.jpg')
img2 =cv2.imread('imgs/IU2.jpg')
img3 =cv2.imread('imgs/IU3.jpg')
img_sh1 =cv2.imread('imgs/shin1.jpg')
img_sh2 =cv2.imread('imgs/shin2.jpg')

# STEP4 : Interface
# faces = app.get(img)
faces1 = app.get(img1)
faces2 = app.get(img2)
faces3 = app.get(img3)
faces_sh1 = app.get(img_sh1)
faces_sh2 = app.get(img_sh2)
print(len(faces1))
print(len(faces2))
print(len(faces3))
print(len(faces_sh1))
print(len(faces_sh2))
# print(faces[0])
# print(faces[0].gender)
# print(faces[0].age)

# STEP5 : Post Processing
rimg = app.draw_on(img1, faces1)
cv2.imwrite("./IU1_output.jpg", rimg)

rimg = app.draw_on(img2, faces2)
cv2.imwrite("./IU2_output.jpg", rimg)

rimg = app.draw_on(img3, faces3)
cv2.imwrite("./IU3_output.jpg", rimg)

rimg = app.draw_on(img_sh1, faces_sh1)
cv2.imwrite("./shin1_output.jpg", rimg)

rimg = app.draw_on(img_sh2, faces_sh2)
cv2.imwrite("./shin2_output.jpg", rimg)

# STEP6 : Application (Face Verification)
# then print all-to-all face similarity
feats = []
# for face in faces:
for face in [faces1[0], faces2[0], faces3[0], faces_sh1[0], faces_sh2[0]]:
    feats.append(face.normed_embedding)
feats = np.array(feats, dtype=np.float32)
sims = np.dot(feats, feats.T)
print(sims)

# STEP7 : Application (Face Verification)
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces_sh2[0].normed_embedding, dtype=np.float32)
sims = np.dot(feat1,feat2)
print(sims)
if sims > 0.55 :
    print('동일인 입니다.')
else:
    print('동일인이 아닙니다.')