from sklearn.preprocessing import maxabs_scale
from sklearn.externals import joblib
from record import recording
import numpy as np
import librosa
import RPi.GPIO as GPIO

# Open recorder
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)

#=============================================================
model=joblib.load("randomforest20.pkl")
print("Finish Load model")
#=============================================================

GPIO.output(4, False)

while(True):
    
    recording()
    #=============================================================
    raw, sr = librosa.load("test.wav")
    norm = maxabs_scale(raw[4100:])
    data = librosa.feature.mfcc(norm, sr=44100, n_mfcc=13).T

       
    prediction = model.predict(data)
    result = (prediction==0).sum() / len(prediction)
    
    if(result > 0.8):
        print("Result", result, "Background")
        # Print for debugging
        GPIO.output(4, False)
    else:
        print("Result", result, "Drone")
        # Print for debugging
        GPIO.output(4, True)
        
    #=============================================================