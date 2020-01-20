import dlib
import cv2
import numpy as np
import math,random,string
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	dim = (h,width)

	# resize the image
	resized = cv2.resize(image, dim, interpolation=inter)

	return resized

def file_name(size=6, chars=string.ascii_uppercase + string.digits):
	return ''.join(random.choice(chars) for _ in range(size))+'.jpg'

detector = dlib.get_frontal_face_detector()  				#dlib's face detector (uses HOG)
predictor = dlib.shape_predictor('facial_landmarks.dat')	#dlib's pretrained model to recognise facial features (eyes,jawline,mouth etc)

cam=cv2.VideoCapture(0)

while True:
	ret,image = cam.read()
	image = resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)			#HOG takes grayscale input

	faces = detector(gray, 1)


	for (i, face) in enumerate(faces):
		x1, y1, x2, y2, w, h = face.left(), face.top()+200, face.right() + 1, face.bottom() + 1, face.width(), face.height()-30
		x1-=10
		x2+=10
		y1-=120
		y2+=40
		cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
		cv2.imwrite(file_name(), image[y1:y2,x1:x2])

	cv2.imshow("Face detection", image)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
