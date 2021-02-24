import tensorflow as tf
from absl import flags
from absl.flags import FLAGS

# flags.DEFINE_integer('size', 416, 'image_size: default is 416')

# defining utility functions for data transformation


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                box_xy = tf.math.abs(box_xy)

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                anchor_idx = anchor_idx  # for negative index only in svhn extra
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)
                grid_xy = grid_xy  # for negative index only in svhn extra

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    # might be a -ve value is producing here. take its absolute value and remove ans operator from everywhere else
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)
    anchor_idx = anchor_idx  # for negative index only in svhn extra

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = tf.cast((x_train / 255.0), tf.float32)
    return x_train


image_feature_description = {
    # in case dimension of image may be useful.
    'height': tf.io.FixedLenFeature([], tf.float32),
    'width': tf.io.FixedLenFeature([], tf.float32),
    'depth': tf.io.FixedLenFeature([], tf.float32),
    'label': tf.io.FixedLenFeature([], tf.string),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

# def _parse_image_function(example_proto):
# Parse the input tf.Example proto using the dictionary above.
# return tf.io.parse_example(example_proto, image_feature_description)


def parse_tfrecord(tfrecord, size):
    features = tf.io.parse_single_example(tfrecord, image_feature_description)
    x_train = tf.image.decode_png(features['image_raw'], channels=3)
    # x_train = tf.cast(x_train,tf.float32)
    # x_trains will be batched before
    x_train = tf.image.resize(x_train, (size, size))
    # mapping

    image_label = features['label']
    image_label = tf.io.parse_tensor(image_label, out_type=tf.float64)
    y_train = tf.cast(image_label, tf.float32)

    paddings = [[0,  FLAGS.yolo_max_boxes -
                 tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(path, size=416):
    dataset = tf.data.TFRecordDataset(
        path, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    return dataset.map(lambda x: parse_tfrecord(x, size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
