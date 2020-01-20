import cv2
import numpy as np
import math

def get_mouth(face,image):
	x1, y1, x2, y2, w, h = face.left(), face.top()+200, face.right() + 1, face.bottom() + 1, face.width(), face.height()-30
	x1-=10
	x2+=10
	y1-=120
	y2+=40
	coords = np.zeros((2, 2), dtype="int")
	# Converts points of interest from (x,y) to [x y] format 
	coords[0] = (x1,y1)
	coords[1] = (x2,y2)
	return coords

def get_eyes(features):
	left_eye=features[36:42]
	right_eye=features[42:48]
	return left_eye,right_eye

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	dim = (h,width)

	# resize the image
	resized = cv2.resize(image, dim, interpolation=inter)

	return resized

def reshape(img, points,size = (24,24),scale=2.3):
	x1, y1 = np.amin(points, axis=0)
	x2, y2 = np.amax(points, axis=0)
	cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

	w = (x2 - x1) * scale
	h = w * size[1] / size[0]

	margin_x, margin_y = w / 2, h / 2
	min_x, min_y = int(cx - margin_x), int(cy - margin_y)
	max_x, max_y = int(cx + margin_x), int(cy + margin_y)

	rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

	out_img = img[rect[1]:rect[3], rect[0]:rect[2]]
	out_img = cv2.resize(out_img, dsize=size)
	out_img = out_img.reshape((1, size[1], size[0], 1)).astype(np.float32) 
	return out_img



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

def distance(a,b):
	x1,y1=a
	x2,y2=b
	return math.sqrt((abs(x1-x2)**2)+(abs(y1-y2)**2))