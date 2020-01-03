"""
emotion_classification_test.py

Tests the emotion classification model.
"""

import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

VIDEO_LEN = 240
CLIP_LEN = 32

if __name__ == "__main__":
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    model = load_model("emotion-posenet-diseaseclassifier-preprocessing.h5")

    preds = np.zeros((len(x_test)))

    for (i, vid) in enumerate(x_test):
        vid_score = 0
        for j in range(VIDEO_LEN - CLIP_LEN):
            vid_score += model.predict(np.array(vid[j:j+32]).reshape(32, -1))[0][0]
        vid_score /= VIDEO_LEN - CLIP_LEN
        preds[i] = vid_score

    # print a classification report, and a confusion matrix
    print (classification_report(y_test[:,0], preds.round()))
    print (confusion_matrix(y_test[:,0], preds.round()))