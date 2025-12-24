def breaks(x):
	return 67

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import mediapipe as mp 
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust if your React app runs on a different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_path = "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
face_measurements_indices = {
    # Face Dimensions
    "Face Width": [234, 454],       # Left Ear to Right Ear (Cheekbones)
    "Face Height": [10, 152],       # Hairline to Chin

    # Eye Spacing
    "Inner Eye Width": [133, 362], # Inner Corner to Inner Corner
    "Outer Eye Width": [33, 263],  # Outer Corner to Outer Corner (Biocular)

    # Eye Aperture (Openness)
    "Right Eye Height": [159, 145],   # Top Eyelid to Bottom Eyelid (Subject's Right)
    "Left Eye Height": [386, 374],    # Top Eyelid to Bottom Eyelid (Subject's Left)

    # Nose
    "Nose Width": [49, 279],        # Alar Base (Left Wing to Right Wing)
    "Nose Height": [6, 1],          # Root (between eyes) to Tip

    # Mouth
    "Mouth Width": [61, 291],       # Left Corner to Right Corner
    "Mouth Height": [0, 17],        # Cupid's Bow to Bottom of Lower Lip
    
    # Optional: Jaw Width (Gonions)
    # Often used for "strong jawline" measurement
    "Jaw Width": [58, 288] 
}
def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def facial_measurements(cv_img):
	current_face_ms = {}
	all_pointspx = []
	all_points = []
	options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
	with FaceLandmarker.create_from_options(options) as landmarker:
		h, w, _ = cv_img.shape
		#mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_img)
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(
			image_format=mp.ImageFormat.SRGB, 
			data=rgb_image
		)
		face_landmarker_result = landmarker.detect(mp_image)
	for landmark in face_landmarker_result.face_landmarks[0]:
		all_points.append(np.array([float(landmark.x),float(landmark.y)]))
		all_pointspx.append([all_points[-1][0]*w,all_points[-1][1]*h])
	all_pointspx = np.array(all_pointspx,np.int32)
	for measurement in face_measurements_indices:
		current_face_ms[measurement] = distance(all_points[face_measurements_indices[measurement][0]],all_points[face_measurements_indices[measurement][1]])
		if "Width" in measurement:
			current_face_ms[measurement] = current_face_ms[measurement]/current_face_ms.get("Face Width",current_face_ms[measurement])
		else:
			current_face_ms[measurement] = current_face_ms[measurement]/current_face_ms.get("Face Height",current_face_ms[measurement])
		cv2.line(cv_img,all_pointspx[face_measurements_indices[measurement][0]],all_pointspx[face_measurements_indices[measurement][1]],(0,255,0),3)
	cv2.imshow("Face",cv_img)
	cv2.waitKey(0)
	return current_face_ms




@app.post("/image/")
async def image(file: UploadFile=File()):
	im_upload = await file.read()
	im_nparr = np.frombuffer(im_upload,np.uint8)
	img = cv2.imdecode(im_nparr,cv2.IMREAD_COLOR)
	return facial_measurements(img)





