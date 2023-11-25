from fastapi import FastAPI, UploadFile

# STEP1 : Install Packages
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP2 : Create Task Processor
# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app = FaceAnalysis(alloed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
face = FaceAnalysis(providers=['CPUExecutionProvider'])
face.prepare(ctx_id=0, det_size=(640, 640))

app = FastAPI()
import io
from PIL import Image

from fastapi.responses import StreamingResponse
import cv2
@app.post('/predict')
async def predict_api(image_file1: UploadFile):
    # 0. read bytes from http
    content1 = await image_file1.read()
    
    # 1. make buffer from bytes
    buffer1 = io.BytesIO(content1)

    # 2. decode image from buffer
    pil_img1 = Image.open(buffer1)

    # STEP3 : Get Image : Pre Processing
    # img =cv2.imread('imgs/twice.jpg')
    image1 = np.array(pil_img1)

    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

    # STEP4 : Interface
    faces1 = face.get(image1)

    # STEP5 : Post Processing
    rimg = face.draw_on(image1, faces1)

    # 6. Encode and return the image
    img_encode = cv2.imencode('.png', rimg)[1]
    image_stream = io.BytesIO(img_encode.tobytes())
    return StreamingResponse(image_stream, media_type="image/png")

@app.post('/predict_img')
async def predict_api_img(image_file1: UploadFile, image_file2: UploadFile):
    # 0. read bytes from http
    content1 = await image_file1.read()
    content2 = await image_file2.read()
    
    # 1. make buffer from bytes
    buffer1 = io.BytesIO(content1)
    buffer2 = io.BytesIO(content2)

    # 2. decode image from buffer
    pil_img1 = Image.open(buffer1)
    pil_img2 = Image.open(buffer2)

    # STEP3 : Get Image : Pre Processing
    image1 = np.array(pil_img1)
    image2 = np.array(pil_img2)

    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    # STEP4 : Interface
    faces1 = face.get(image1)
    faces2 = face.get(image2)

    # STEP5 : Post Processing
    # rimg = app.draw_on(image1, faces1)
    # rimg = app.draw_on(image2, faces2)


    # STEP6 : Application (Face Verification)
    # then print all-to-all face similarity
    feats = []
    # for face in faces:
    for faces in [faces1[0], faces2[0]]:
        feats.append(faces.normed_embedding)
    feats = np.array(feats, dtype=np.float32)
    sims = np.dot(feats, feats.T)
    print(sims)

    # STEP7 : Application (Face Verification)
    feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
    feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
    sims = np.dot(feat1,feat2)
    if sims > 0.55 :
        result = '동일인 입니다.'
    else:
        result = '동일인이 아닙니다.'
    return{'result' : result}