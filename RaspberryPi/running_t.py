from sklearn.preprocessing import maxabs_scale
from record import recording
import tensorflow as tf
import numpy as np
import librosa
import RPi.GPIO as GPIO

# Open recorder
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)

#=============================================================
# Read Model
## Model parameter 
sd = 1 / np.sqrt(13)

## Model
X = tf.placeholder(tf.float32,[None,13])
Y = tf.placeholder(tf.float32,[None,4])

W_1 = tf.Variable(tf.random_normal([13,280], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([280], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([280,300], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([300], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W = tf.Variable(tf.random_normal([300,4], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([4], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)                  #multi classification:softmax

init = tf.global_variables_initializer()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1])) 
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost_function)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Model loader
saver = tf.train.Saver()

cost_history = np.empty(shape=[1],dtype=float)
y_pred = None
init = tf.global_variables_initializer()
print("Finish model setting")
#=============================================================

GPIO.output(4, True)

while(True):
    
    recording()
    #=============================================================
    raw, sr = librosa.load("test.wav")
    norm = maxabs_scale(raw[4100:])
    x = librosa.feature.mfcc(norm, sr=44100, n_mfcc=13).T

    with tf.Session() as sess:
        saver.restore(sess, "/home/pi/nn15/nn_e15")
        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: x})    

    result = (y_pred==0).sum() / len(y_pred)
    
    if(result > 0.9):
        print("Result", result, "Background")
        # Print for debugging
        GPIO.output(4, True)
    else:
        print("Result", result, "Drone")
        # Print for debugging
        GPIO.output(4, False)
        
    #=============================================================