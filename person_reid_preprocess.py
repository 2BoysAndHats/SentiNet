"""
person_reid_preprocess.py

Takes in the Market-1501 dataset, and combines all the images into a numpy file by pedestrian (each gets their own ID)
Also removes all distractor and junk images (for now?)
"""

import glob
import cv2
import numpy as np
from os.path import basename

BASE_DIR = 'Market-1501/'
TRAIN_DIR = BASE_DIR + 'bounding_box_train/'
TEST_DIR = BASE_DIR + 'bounding_box_test/'

def process (path_list):
    pedestrians = {}

    for path in path_list:
        # grab the pedestrian ID
        path_split = basename(path[:-4]).split('_')
        id = int(path_split[0])

        # is this a bogus frame?
        if id == 0 or id == -1:
            continue

        # append this frame to the list
        if id not in pedestrians:
            pedestrians[id] = []

        pedestrians[id].append(path)

    dataset = [[cv2.imread(i) for i in pedestrian] for pedestrian in pedestrians.values()]

    return dataset


def main ():
    # get a list of all the photos we're working with
    train_photos = glob.glob(TRAIN_DIR + '*.jpg')
    test_photos = glob.glob(TEST_DIR + '*.jpg')

    train = process (train_photos)
    test = process (test_photos)

    np.savez_compressed("images_by_pedestrian_train", train)
    np.savez_compressed("images_by_pedestrian_test", test)

if __name__ == "__main__":
    main()