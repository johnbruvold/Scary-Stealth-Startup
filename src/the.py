def breaks(x):
	return 67

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust if your React app runs on a different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/image/")
async def image(file: UploadFile=File()):
	im_upload = await file.read()
	im_nparr = np.frombuffer(im_upload,np.uint8)
	img = cv2.imdecode(im_nparr,cv2.IMREAD_COLOR)
	cv2.imshow("user_img",img)
	cv2.waitKeyEx(1)
	return {
			"width":img.shape[1],
			"height":img.shape[0],
			"image":img
			}





