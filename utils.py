'''
I will define methods to load pretrained weights into Darknet layers('the only case where we can load darknet weights and which are compatible with variable number of classes.'),
and few methods to draw labels and boxes for later use.
'''

from absl import logging
import tensorflow as tf
import numpy as np
import cv2

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i+1 < len(sub_model.layers) and sub_model.layers[i+1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i+1]

            logging.info("{}/{} {}".format(sub_model.name,
                                           layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, np.float32, count=filters)
            else:
                # darknet β,Ɣ, mean(µ), variance(𝞼²)
                bn_weights = np.fromfile(wf, np.float32, count=4*filters)
                # tf β,Ɣ, mean(µ), variance(𝞼²)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

             # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])

    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

        COLORS = [np.random.randint(0, 255), np.random.randint(
            0, 255), np.random.randint(0, 255)]
        # COLORS = [0, 255, 0]
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1

        (text_width, text_height) = cv2.getTextSize('{} {:.2f}'.format(
            class_names[int(classes[i])], objectness[i]), font, fontScale=font_scale, thickness=1)[0]

        text_offset_x = x1y1[0]
        text_offset_y = x1y1[1]
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x +
                                                       text_width + 2, text_offset_y - text_height - 2))

        img = cv2.rectangle(img, x1y1, x2y2, COLORS, 1)
        img = cv2.rectangle(img, box_coords[0], box_coords[1], COLORS, -1)
        img = cv2.putText(img, '{} {:.2f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 1)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 1)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
