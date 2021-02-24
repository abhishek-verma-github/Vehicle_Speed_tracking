import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import os

base_directory = '/Users/abhishekverma/Desktop/python-env/NeuralNet examples/YOLO_digit'

# train_label.head(5)


def get_size(label_df, directory, i):
    index = int(label_df['filename'][i].split('.')[0])
    img = Image.open(directory+f'/{index}.png')
    w, h = img.size
    return w, h

# directory = base_directory+'/train'
# w,h = get_size(directory,0)
# print(f'width: {w}, height: {h}')


def encode_y(label, len_, image_directory):
    # label--> label_DataFrame
    Y = []
    for i in range(len_):
        directory = f'{base_directory}/{image_directory}'
        w, h = get_size(label, directory, i)
        # (x1,y1,x2,y2)
        m = np.array([[d['left']/w, d['top']/h, ((d['left']+d['width'])/(2*w)),
                       ((d['top']+d['height'])/(2*h))] for d in label['boxes'][i]]).astype(np.float32)
        classes = np.array([[d['label']-1]  # label are 1 less than in actual label--> for cross entropy... also this is the way they should be labelled for loss calculation
                            for d in label['boxes'][i]]).astype(np.float32)
        y = np.concatenate([m, classes], axis=-1)
        if y.shape[0] < 6:
            p_dim = 6 - y.shape[0]
            # 5 = x1,y1,x2,y2,class_index{starts from 0}
            p = np.zeros((p_dim, 5))
            y = np.concatenate((y, p), axis=0)
        if i > 0 and Y[i-1].shape != y.shape:
            print('not equal', f'm{i}.shape : {y.shape}')
        Y.append(y)  # returning a list
    return Y


def create_image_label_mapping(glob_image, encoded_y, label_df):
    """ returns a dictionary X_train(image_path to open while writing image), Y_label
        to be unpacked for writing into TFRecord """
    image_labels = dict()
    for image in glob_image:
        image_idx = int(os.path.basename(
            os.path.normpath(image)).split('.')[0])
        label_idx = int(label_df['filename'][image_idx - 1].split('.')[0]) - 1
        image_labels[image] = encoded_y[label_idx]
    return image_labels


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    '''Since we will use an array in label feature, 
        and _bytes_feature_ and _float_feature_ accept only one value tp serialize. 
        thats why we will serialize array on our own
    '''
    array = np.array(array, dtype=np.float64)
    array = tf.io.serialize_tensor(array)
    return array


def serialize_example(image_string, label):
    image_shape = tf.image.decode_png(image_string).shape
    serialized_label = serialize_array(label)
    feature = {
        'height': _float_feature(image_shape[0]),
        'width': _float_feature(image_shape[1]),
        'depth': _float_feature(image_shape[2]),
        'label': _bytes_feature(serialized_label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


### training dataset ###
train_label = pd.read_json(base_directory + '/svhn_train.json')
y_train = encode_y(train_label, len(train_label), image_directory='train')
images = sorted(glob.glob(f'{base_directory}/train/*.png'))
image_labels = create_image_label_mapping(images, y_train, train_label)


# Write the raw image files to `images.tfrecords`.
# First, process the two images into `tf.Example` messages.
# Then, write to a `.tfrecords` file.
record_file = 'trainlabels.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        # process the image and label into `tf.Example` messages(proto)
        tf_example = serialize_example(image_string, label)
        writer.write(tf_example.SerializeToString())

# Read the TFRecord file---->
# we now have the file—trainalabels.tfrecords—and can now iterate over the records in it
# to read back what we wrote


# Create a dictionary describing the features
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.float32),
    'width': tf.io.FixedLenFeature([], tf.float32),
    'depth': tf.io.FixedLenFeature([], tf.float32),
    # since we serialized label array as byte string jus as image
    'label': tf.io.FixedLenFeature([], tf.string),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_example(example_proto, image_feature_description)


#raw_image_dataset = tf.data.TFRecordDataset('trainlabels.tfrecords')
# parsed_image_dataset = raw_image_dataset.map(_parse_image_function)


# from IPython import display
# for image_features in parsed_image_dataset.take(5):
#     image_label = image_features['label'].numpy()
#     image_raw = image_features['image_raw'].numpy()
#     image_label = tf.io.parse_tensor(image_label, out_type=tf.float64)
#     display.display(display.Image(data=image_raw))
#     print(image_label)


### Now we will create TFRecord for validation ###

val_label = pd.read_json(base_directory + '/svhn_test.json')
y_val = encode_y(val_label, len(val_label), image_directory='test')
val_images = sorted(glob.glob(
    '/Users/abhishekverma/Desktop/python-env/NeuralNet examples/YOLO_digit/test/*.png'))
val_image_labels = create_image_label_mapping(val_images, y_val, val_label)
record_file = 'validation.tfrecords'

with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in val_image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = serialize_example(image_string, label)
        writer.write(tf_example.SerializeToString())

# raw_val_image_dataset = tf.data.TFRecordDataset('validation.tfrecords')
# parsed_val_image_dataset = raw_val_image_dataset.map(_parse_image_function)

# for image_features in parsed_val_image_dataset.take(5):
#     image_label = image_features['label'].numpy()
#     image_raw = image_features['image_raw'].numpy()
#     image_label = tf.io.parse_tensor(image_label, out_type=tf.float64)
#     display.display(display.Image(data=image_raw))
#     print(image_label)
