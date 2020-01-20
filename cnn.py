
from keras.models import load_model

def get_prediction_value(model,cnn_input):
    prediction=model.predict(cnn_input)
    return prediction

def predict_eye(model,left_eye,right_eye):
    left_eye_prediction=get_prediction_value(model,left_eye)
    right_eye_prediction=get_prediction_value(model,right_eye)

    prediction=(left_eye_prediction+right_eye_prediction)/2.0

    if prediction>0.5:
        prediction="open"
    else:
        prediction="close"
    
    return prediction

def predict_mouth(model,mouth):
    prediction=get_prediction_value(model,mouth)
    if prediction>0.5:
        prediction="close"
    else:
        prediction="open"
    
    return prediction
