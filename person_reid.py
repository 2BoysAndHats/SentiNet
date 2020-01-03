"""
person_reid.py

Trains the person reidentification model, and implements rank-n testing.
"""

import keras
import keras.backend as K

from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet

from keras.layers import Input, Dense, Lambda, Flatten
from keras.models import Model
from keras.optimizers import Adam

import numpy as np

def load_data():
    # load in the pre-processed data
    train = np.load("images_by_pedestrian_train.npz", allow_pickle = True)['arr_0']
    test = np.load("images_by_pedestrian_test.npz", allow_pickle = True)['arr_0']

    return (train, test)

# now, make pairs for training - half the same, half different.
def get_pairs(features, batch_size = 100):
    # returns an array of pairs, half identical, half not.
    num_classes = len(features)
    
    pairs = [np.zeros((batch_size, 128, 64, 3)) for i in range (2)]
    targets = np.ones((batch_size,))
    
    for i in range (batch_size):
        class_a = np.random.randint(num_classes)
        class_b = class_a
        
        if i > batch_size / 2:
            # different class
            class_b = np.random.randint(num_classes)
            targets[i] = 0
            
        # now pick a random sample from each class
        pairs[0][i] = features[class_a][np.random.randint(len(features[class_a]))]
        pairs[1][i] = features[class_b][np.random.randint(len(features[class_b]))]
        
    return pairs, targets

# and a generator to turn out as many of the batches as we want!
def generate(features, batch_size = 100):
    """
    a generator for batches, so model.fit_generator can be used.
    """
    while True:
        pairs, targets = get_pairs(features, batch_size)
        yield (pairs, targets)

def make_train_model():
    # model time!
    feature_net = MobileNet(include_top = False, input_shape = (128, 64, 3), weights = 'imagenet')

    left_input = Input((128, 64, 3))
    right_input = Input((128, 64, 3))

    # run the input images through our feature extractor
    left_features = feature_net(left_input)
    right_features = feature_net(right_input)

    # and flatten them to a feature vector
    left_features_flat = Flatten()(left_features)
    right_features_flat = Flatten()(right_features)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([left_features_flat, right_features_flat])
        
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)
        
    # Connect the inputs with the outputs
    model = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam(lr = 0.00006)
    model.compile(loss="binary_crossentropy", metrics = ['acc'], optimizer=optimizer)

    return model

def make_test_model():
    feature_input_l = Input(shape = (8192,))
    feature_input_r = Input(shape = (8192,))

    lambdad = model.get_layer("lambda_1")([feature_input_l, feature_input_r])
    dense_output = model.get_layer("dense_1")(lambdad)

    return Model(inputs = [feature_input_l, feature_input_r], outputs = dense_output)

def extract_features (data):
    pedestrian_features = []
    for ped in data:
        features = feature_net.predict(np.array(ped))
        features = np.reshape(features, (len(ped), -1))
        pedestrian_features.append(features)
            
    return np.array(pedestrian_features)

def top_n_acc(n, test_features, compari_net):
    hits = 0

    # comparison images - just average 
    comparison_images = np.array([sum(i) / len(i) for i in test_features])

    for (i, person) in enumerate(test_features):
        
        output = None
        
        for _ in range (16):
            # pick a random query image of the person we're testing
            query_image = [person[np.random.randint(len(person))]] * len(comparison_images)
        
            # run it through the model
            if output is None:
                output = compari_net.predict([query_image, comparison_images]).reshape((-1))
            else:
                output += compari_net.predict([query_image, comparison_images]).reshape((-1))
        
        ranked_people = (-output).argsort(axis=0)
        
        # is it in the top n?
        top_n = ranked_people[:n]
        
        if i in top_n:
            hits += 1
    
    return hits / len(test_features)

if __name__ == "__main__":
    # load our train and test data
    print ("Loading data ... ", end = "")
    train, test = load_data()
    print ("data loaded!")


    print ("Loading model ... ", end = "")
    model = make_train_model()
    print ("model loaded!")

    print (model.summary())

    print ("Begin training")
    model.fit_generator(
        generate(train),
        steps_per_epoch = 1000,
        epochs = 4,
    )

    model.save("person_reid.h5")

    print ("Start testing. Preprocessing test features ... ", end = "")
    # preprocess our test features - we're going to be using them a lot
    # so it makes sense just to pass them through the featurenet once
    test_features = extract_features(test)
    print ("extracted.")

    # extract a test model - just the train model without the featurenet (we don't need it any more)
    print ("Making test model ... ", end = "")
    test_model = make_test_model()
    print ("model extracted.")

    print ("Begin accuracy testing:")
    RANK_N = [1, 5] # what rank scores are we looking to find?
    for n in RANK_N:
        score = top_n_acc(n, test_features, test_model)
        print (f"\tTop-{n} accuracy: {score * 100:.2f}%")