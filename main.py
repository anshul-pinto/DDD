import dlib,time
from helper import *
from ear_calculate import *
from cnn import *
from sklearn.externals import joblib

detector = dlib.get_frontal_face_detector()  				#dlib's face detector (uses HOG)
predictor = dlib.shape_predictor('facial_landmarks.dat')	#dlib's pretrained model to recognise facial features (uses regression trees)
eye_model = load_model('drowsyv3.hd5')						#Trained model for predicting state of the eyes
mouth_model = load_model('yawn2.hd5')						#Trained model for predicting state of the mouth
ear_model = joblib.load('tree_ear.pkl')

def detect_drowsy(eyes,decision_parameter=0.7):
	driver_state = ""	
	eye_in_favour=eyes.count("close")
	if eye_in_favour/len(eyes)>=decision_parameter:
		driver_state = "drowsy eyes"
	return driver_state

cam=cv2.VideoCapture(0)
eye_history_ear=[]
eye_history_cnn=[]
mouth_history_cnn=[]
cnn_drowsy=""
ear_drowsy=""

while True:
	
	ret,image = cam.read()
	image = resize(image, width=500)
	cnn_image=image.copy()
	#print(cnn_image)
	gray = cv2.cvtColor(cnn_image, cv2.COLOR_BGR2GRAY)				#Gray input for CNN
	brighter_image = increase_brightness(image)
	equalized_image = histogram_equalization(brighter_image)
	faces = detector(equalized_image, 1)

	for (i, face) in enumerate(faces):
		facial_features = predictor(equalized_image, face)
		facial_features = shape_to_np(facial_features)		
		
		#EAR
		ear=calculate_ear(facial_features)
		if ear_model.predict(np.reshape(ear,(1,-1)))==0:
			eye_state_ear="close"
		else:
			eye_state_ear="open"

		eye_history_ear.append(eye_state_ear)

		if len(eye_history_ear)>=15:
			ear_drowsy=detect_drowsy(eye_history_ear)
			eye_history_ear=[]

		#CNN - Eye
		left_eye_ear,right_eye_ear = get_eyes(facial_features) 	
		right_eye_cnn = reshape(gray, right_eye_ear)
		left_eye_cnn = reshape(gray, left_eye_ear)
		eye_state_cnn = predict_eye(eye_model,left_eye_cnn,right_eye_cnn)

		eye_history_cnn.append(eye_state_cnn)

		if len(eye_history_cnn)>=15:
			cnn_drowsy=detect_drowsy(eye_history_cnn)
			eye_history_cnn=[]


		cv2.putText(cnn_image, eye_state_cnn, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
		cv2.putText(cnn_image, cnn_drowsy, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
		cv2.putText(image, eye_state_ear, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
		cv2.putText(image, ear_drowsy, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
	
	
	cv2.imshow("EAR", image)
	
	cv2.imshow("CNN", cnn_image)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()