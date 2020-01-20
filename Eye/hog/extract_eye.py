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

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray) 

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def rect_to_bb(rect):
	# Converts the bounding box predicted by dlib to the OpenCv's (x, y, w, h) format
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# Converts points of interest from (x,y) to [x y] format 
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords


def file_name(size=6, chars=string.ascii_uppercase + string.digits):
	return ''.join(random.choice(chars) for _ in range(size))+'.jpg'

def distance(a,b):
	x1,y1=a
	x2,y2=b
	return math.sqrt((abs(x1-x2)**2)+(abs(y1-y2)**2))

def calculate_ear(eye):
	#Formula given in https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
	ear = ( distance(eye[1],eye[5]) + distance(eye[2],eye[4]) ) / (2*distance(eye[0],eye[3]))
	return ear

detector = dlib.get_frontal_face_detector()  				#dlib's face detector (uses HOG)
predictor = dlib.shape_predictor('facial_landmarks.dat')	#dlib's pretrained model to recognise facial features (uses regression trees)

cam=cv2.VideoCapture(0)

while True:
	ret,image = cam.read()
	image = resize(image, width=500)
	brighter_image = increase_brightness(image)
	equalized_image = histogram_equalization(brighter_image)
	faces = detector(equalized_image, 1)


	for (i, face) in enumerate(faces):
		facial_features = predictor(equalized_image, face)
		facial_features = shape_to_np(facial_features)


		left_eye=facial_features[36:42]
		right_eye=facial_features[42:48]
		for (x, y) in left_eye:
			x1, y1, x2, y2, w, h = x, y, x+ 1, y + 1,1,1
			x1-=20
			x2+=40
			y1-=50
			y2+=10
			#cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
			cv2.imwrite("LEFT-"+file_name(), image[y1:y2,x1:x2])
			break
		for (x, y) in right_eye:
			x1, y1, x2, y2, w, h = x, y, x+ 1, y + 1,1,1
			x1-=20
			x2+=40
			y1-=50
			y2+=10
			#cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
			cv2.imwrite("RIGHT-"+file_name(), image[y1:y2,x1:x2])
			break
	
	cv2.imshow("Eye detection", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
