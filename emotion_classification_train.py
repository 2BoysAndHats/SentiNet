"""
emotion_classification_train.py

Conor Browne

This file takes in the  EWalk dataset, preprocesses it by resizing and normalizing the frames, and then trains and saves the emotion classification model.
"""

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Conv1D, BatchNormalization, ReLU, AveragePooling1D, Softmax, Input, Flatten, Reshape
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from collections import Counter
import random
import os
import gc

random.seed(123) # repeatability

# define some paths
INPUT_PATH = "ewalk-data/"

CMU_VIDEOS = INPUT_PATH + "cmu/allavi/subjects/"
CMU_LABELS_PATH = INPUT_PATH + "cmu/ParticipantResponses.csv"
CMU_EXT = ".avi"

EWALK_VIDEOS = INPUT_PATH + "ewalk/Videos/"
EWALK_LABELS_PATH = INPUT_PATH + "ewalk/ParticipantResponses.csv"
EWALK_EXT = ".mp4"

NUM_CLASSES = 2 # happy, or sad
VIDEO_LEN = 240 # what's the longest chunk of video we'll store?

def process_labels_to_score (labels_df):    
    label_scores = {}
    
    for column in labels_df.columns[1:]:    
        # grab the average score
        emo_score = labels_df[column].iloc[1:].dropna().astype(int).mean()
        
        base_id = (column if "." not in column else column.split(".")[0])
        emotion_id = int(0 if "." not in column else column.split(".")[1])
        
        if base_id not in label_scores:
            label_scores[base_id] = {}
            
        label_scores[base_id][emotion_id] = emo_score
        
    return label_scores

def filter_label_score (scored_labels):
    classified_emotions = {}
    
    for (label, scores) in scored_labels.items():
        
        scores_vals = list(scores.values())
        
        if scores_vals[2] >= 3.5: # is it a strong sad?
            classified_emotions[label] = 1
        else:
            classified_emotions[label] = 0
        
    return classified_emotions
        
# merge the processed labels, adding on the full path and extension
def merge_labels(tuples):
    final_labels = {}
    
    for (basepath, clips, ext, cmu) in tuples:
        for (clip, emotion) in clips.items():
            path = basepath + clip.strip() + ext
            if cmu:
                name = clip.strip()
                path = basepath + name.split("_")[0] + '/' + name + ext
                
            final_labels[path] = emotion
            
    return final_labels

def load_labels():
    # load the labels
    cmu_labels = pd.read_csv(CMU_LABELS_PATH)
    ewalk_labels = pd.read_csv(EWALK_LABELS_PATH)

    ewalk_scored_labels = process_labels_to_score(ewalk_labels)
    ewalk_labels_processed = filter_label_score(ewalk_scored_labels)

    cmu_scored_labels = process_labels_to_score(cmu_labels)
    cmu_labels_processed = filter_label_score(cmu_scored_labels)

    labels_processed = merge_labels(((EWALK_VIDEOS, ewalk_labels_processed, EWALK_EXT, False), (CMU_VIDEOS, cmu_labels_processed, CMU_EXT, True)))
    labels_processed = list(labels_processed.items())
    random.shuffle(labels_processed)

    return labels_processed

def import_posenet_model(sess):
    cwd = os.getcwd()
    os.chdir("posenet-python")
    import posenet
    os.chdir(cwd)

    return posenet.load_model(100, sess)[1][0] # return heatmap

def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def get_most_common_res(labels_processed):
    # find the most common resolution amongst the input videos
    resolutions = []

    for (path, _) in labels_processed:
        cap = cv2.VideoCapture(path)
        _, frame = cap.read()
        if frame is not None:
            resolutions.append(frame.shape)
            
    return Counter(resolutions).most_common(1)[0][0] # what's the most common item

def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        COMMON_RES[1] * scale_factor, COMMON_RES[0] * scale_factor, output_stride=output_stride)
    
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0 # normalize to [-1, 1]
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale

def posenetify(img, sess, heatmap):
    img_processed, _, scale = _process_input(img)
    return sess.run(heatmap, {'image:0': img_processed}).flatten()

def get_feature_shape(sess, heatmap):
    # calculate flattened feature length
    blank = np.zeros(COMMON_RES, dtype = np.float32)
    return posenetify(blank, sess, heatmap).shape

def load_videos_and_labels(labels_processed, feature_shape, sess, heatmap):
    # how many do we have?
    num_videos = len(labels_processed)

    # load the videos! all the videos!
    video_data = np.zeros((num_videos, VIDEO_LEN) + feature_shape, dtype = np.uint8)
    labels = np.zeros((num_videos, NUM_CLASSES)) # one hot encoding
    
    for (clip_i, (clip, emotion)) in enumerate(labels_processed):
        cap = cv2.VideoCapture(clip)
        frame_i = 0
        ret = True
        clip = []
        
        while ret:
            ret, frame = cap.read()
        
            if frame is None:
                break

            video_data[clip_i, frame_i, :] = np.uint8(posenetify(frame, sess, heatmap))
        
            frame_i += 1
        
            if frame_i == VIDEO_LEN:
                break

        labels[clip_i] = to_categorical(emotion, num_classes = NUM_CLASSES)
        
    return video_data, labels


# ****************
#   MODEL STUFF
# ****************

def data_batch(x, y, feature_shape, batch_size = 8, clip_len = 32):
    # pick some random indices, and try to keep the labels in a 50 / 50 proportion
    a = np.random.choice(np.where(y == (0, 1))[0], size = batch_size // 2)
    b = np.random.choice(np.where(y == (1, 0))[0], size = batch_size // 2)
    video_indices = np.concatenate((a, b))
    np.random.shuffle(video_indices)
    
    videos = x[video_indices]
    
    # and some random starting indices
    starting_indices = np.random.randint(VIDEO_LEN - 32, size = batch_size)
    
    # grab the clips, and run them through posenet
    clips = np.zeros((batch_size, clip_len) + feature_shape)
    
    for (i, (video, indice)) in enumerate(zip(videos, starting_indices)):
        clips[i, :] = video[indice: indice + clip_len]
    
    return (clips, y[video_indices])

def data_gen (x, y, feature_shape, batch_size = 8, clip_len = 32):
    while True:
        yield data_batch(x, y, feature_shape, batch_size, clip_len)

def make_model(feature_shape):
    input_layer = Input(shape = (32,) + feature_shape)

    conv_1 = Conv1D(128, 8)(input_layer)
    batch_norm_1 = BatchNormalization()(conv_1)
    block_1 = ReLU()(batch_norm_1)

    conv_2 = Conv1D(256, 5)(block_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    block_2 = ReLU()(batch_norm_2)

    pool_1 = AveragePooling1D()(block_2)
    flatten_1 = Flatten()(pool_1)
    dense_1 = Dense(NUM_CLASSES)(flatten_1)
    out = Softmax()(dense_1)

    model = Model(inputs=input_layer, outputs = out)
    model.compile("adam", loss = "categorical_crossentropy", metrics = ["acc"])

    return model

if __name__ == "__main__":
    labels_processed = load_labels()
    COMMON_RES = get_most_common_res(labels_processed)

    sess = tf.Session()
    heatmap = import_posenet_model(sess)    

    feature_shape = get_feature_shape(sess, heatmap)

    X, Y = load_videos_and_labels(labels_processed, feature_shape, sess, heatmap)

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        shuffle = False
    )

    np.save("x_train.npy", x_train)
    np.save("x_test.npy", x_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    del X, Y, x_test, y_test
    [gc.collect() for i in range(10)]

    model = make_model(feature_shape)

    print (model.summary())

    model.summary()

    model.fit_generator(
        data_gen(x_train, y_train, feature_shape),
        steps_per_epoch = 512,
        epochs = 6
    )

    model.save("emotion-posenet-diseaseclassifier-preprocessing.h5")