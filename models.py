'''

"You only look once (YOLO) is a state-of-the-art, real-time object detection system.
On a Pascal Titan X it processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev."
                                                                                - Darknet
->HOW IT WORKS?
 Yolo apply a single neural network to the full image.
 This network divides the image into regions and predicts bounding boxes
 and probabilities for each region.
 These bounding boxes are weighted by the predicted probabilities.

YOLOv3 uses a few tricks to improve training and increase performance,
including: multi-scale predictions, a better backbone classifier, and more.
The full details are in yolov3 paper!(https://pjreddie.com/media/files/papers/YOLOv3.pdf)

___________________________________________________________________________________________

-------------------------------------------------------------------------------------------
In this file I will implement the YOLOv3(daknet53) <using tf.keras API> as well as its smaller
version, YOLOv3Tiny for use in smaller applications where running full YOLOv3 is
unfeasible due to system limitations.

'''

from absl import flags, logging
from absl.flags import FLAGS

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
    Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)


flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes in a single image')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou_threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score_threshold')

# we will define 9 anchor_boxes 3 for each of 3 scales and masks for each scale...
# these anchor boxes are not random they are found by K-mean on dataset we are
# interested in. We will here use anchor_boxes derived form mscoco datset by k-mean clustering
#
# NOTE: for raining on custom dataset apply k-mean on width and height and obtain 9 clusters
#       and divide them in group of 9 for each of the three scale outputs.

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416  # <-----------
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

# for yolov3tiny we will use 6 anchors for 2 scales
yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416  # <------------
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

# we divided by 416, since all images were resized to 416X416
# so to obtain anchors on a common scale

# Lets define DarknetConv, instead of applying Conv2D directly we will perform some
# paddings on top-left (as described by author in their paper)


# mish
def Mish():
    ''' Since YOLOv4 came out with a new activation which outperforms ReLU, LeakyReLU
        and even google brain's-Swish activation. so we are going to implement that actiavtion along withof one used in YOLOv3 papers and will see if it improves the Yolov3 as well.
        Mish Activation --> Mish(z) = z.tanh(𝛇(z)) where 𝛇(z) = ln(1+exp(x))
    '''
    def mish(x):
        # m = x*tf.math.tanh(tf.nn.softplus(x))
        return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)
    return mish


def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        # ZeroPadding2D(((top_pad, bottom_pad), (left_pad, right_pad)))(input)
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        # LeakyReLU = if x > 0 -> x otherwise-> ⍺ * x
        x = LeakyReLU(alpha=0.1)(x)  # TODO:Mish()(x) <--use mish activation
    return x

# Now make a residual layer for skip connection


def DarknetResidual(x, filters):
    shortcut = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([shortcut, x])
    return x
# Now define Darknetblocks as described in papers each darknetresidual is repeated
# several number of times and blocked to together as: 1X->2X->8X->8X->4X


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

# Now we will implement Darknet53 with help of layers written above:


def Darknet(name=None):
    x = inputs = Input(shape=[None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)

    # FeatureVector1 -> for small scale detection # x_skip
    x = x_36 = DarknetBlock(x, 256, 8)
    # FeatureVector2 -> for medium scale detection # x_skip
    x = x_61 = DarknetBlock(x, 512, 8)
    # FratureVector3 -> for large scale detection
    x = DarknetBlock(x, 1024, 4)

    return Model(inputs, [x_36, x_61, x], name=name)


# Now after defining Darknet we will implement Detector which is a combination of
# Upsampling, Concatenation and Feature building.
# refer to the diagram yoloconv in project directory.


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(shape=x_in[0].shape[1:]), Input(
                shape=x_in[1].shape[1:])
            # since x_in.shape[0] is batch_size
            x, x_skip = inputs
            # x is next FeatureVector obtained from Darknet
            # x_skip is current FeatureVector

            # Concatenate x with x_skip
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(shape=x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters*2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters*2, 3)
        x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv

# Now make final output model layer for YOLOv3 which outputs
# (Batch,grid,grid,anchors,(tx,ty,tw,th,objectness,one-hot-classes))


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(shape=x_in.shape[1:])
        x = DarknetConv(x, filters*2, 3)

        x = DarknetConv(x, anchors*(5+classes), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, shape=(-1, tf.shape(x)[1],
                                                  tf.shape(x)[2], anchors, classes+5)))(x)
        # (batch,g,g,anchors,classes+5)
        return Model(inputs, x, name=name)(x_in)
    return yolo_output

# define function which takes input(yolo_outputs) outputs boxes, scores, classes
# and valid detections


def yolo_boxes(pred, anchors, classes):
    # pred:(batch_size,grid,grid,anchors,(tx,ty,tw,th,obj,one-hot-classes))
    grid_size = tf.shape(pred)[1]
    tx_ty, tw_th, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    # bx = (𝞼(tx) + Cx)/grid_size ... Normalized x
    # by = (𝞼(ty) + Cy)/grid_size ... Normalized y

    box_xy = tf.sigmoid(tx_ty)
    objectness = tf.sigmoid(objectness)  # obj is a probability
    class_probs = tf.nn.softmax(class_probs)

    # original tx,ty,tw,th for loss
    predicted_box = tf.concat([box_xy, tw_th], axis=-1)

    # create grid: grid[x][y] == (row,column)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(
        grid_size))  # (grid_size,grid_size)
    grid = tf.stack(grid, axis=-1)  # (grid_size,grid_size,2)
    grid = tf.expand_dims(grid, axis=2)  # (grid_size,grid_size,1,2)

    # add Cx,Cy to bx,by i.e., box_xy to get normalized absolute coordinates x and y
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        (tf.cast(grid_size, tf.float32))

    # bw = Pw * exp(tw) ; Pw is prior width
    # bh = Ph * exp(th) ; Ph is prior height
    box_wh = tf.exp(tw_th) * anchors

    box_x1y1 = box_xy - box_wh/2  # xmin,ymin
    box_x2y2 = box_xy + box_wh/2  # xmax,ymax

    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, predicted_box


def broadcast_iou(box_1, box_2):
    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])

    union_area = box_1_area + box_2_area - int_area
    iou = int_area / union_area

    # enclose_left_up = tf.minimum(box_1[..., :2], box_2[..., :2])
    # enclose_right_down = tf.maximum(box_1[..., 2:], box_2[..., 2:])
    # enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    # enclose_area = enclose[..., 0] * enclose[..., 1]
    # giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    # return giou
    return iou


# Non-max suppression for 'Detection'; we will use tf.image.combined_non_max_suppression
# It Greedily selects a subset of bounding boxes in descending order of score.
'''
tf.image.combined_non_max_suppression(boxes, scores,
                                             max_output_size_per_class,
                                             max_total_size, iou_threshold=0.5,
                                             score_threshold=float('-inf'),
                                             pad_per_class=False, clip_boxes=True, name=None
                                     )
This operation performs non_max_suppression on the inputs per batch, across all classes.
Prunes away boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.Bounding boxes are supplied as [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners and the coordinates can be provided as normalized (i.e., lying in the interval [0, 1]) or absolute. Note that this algorithm is agnostic to where the origin is in the coordinate system. Also note that this algorithm is invariant to orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system result in the same boxes being
selected by the algorithm. The output of this operation is the final boxes, scores and classes tensor returned after performing non_max_suppression.

boxes:  A 4-D float Tensor of shape [batch_size, num_boxes, q, 4]. If q is 1 then same boxes
        are used for all classes otherwise, if q is equal to number of classes, class-specific boxes are used.
scores: A 3-D float Tensor of shape [batch_size, num_boxes, num_classes] representing a single
        score corresponding to each box (each row of boxes).
max_output_size_per_class: A scalar integer Tensor representing the maximum number of boxes to
                           be selected by non-max suppression per class
max_total_size:	A scalar representing the maximum number of boxes retained over all classes.

Returns
'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor containing the non-max suppressed boxes.
'nmsed_scores': A [batch_size, max_detections] float32 tensor containing the scores for the boxes.
'nmsed_classes': A [batch_size, max_detections] float32 tensor containing the class for boxes.
'valid_detections': A [batch_size] int32 tensor indicating the number of valid detections per batch item. Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid.
The rest of the entries are zero paddings.

'''


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, confidences, class probs
    b, c, t = [], [], []

    # outputs is a tuple of outputs from three scales. o/p-1, o/p-2,o/p-3
    # for all three o/p's we append boxes,confidences and type
    for output in outputs:

        # logging.info(f'outout_shape: {tf.shape((output[0])[0])}')

        b.append(tf.reshape(output[0], shape=(tf.shape(output[0])[0], -1,
                                              tf.shape(output[0])[-1])))
        c.append(tf.reshape(output[1], shape=(tf.shape(output[1])[0], -1,
                                              tf.shape(output[1])[-1])))
        t.append(tf.reshape(output[2], shape=(tf.shape(output[2])[0], -1,
                                              tf.shape(output[2])[-1])))

    # logging.info(f'b: {b}')

    bbox = tf.concat(b, axis=1)

    confidence = tf.concat(c, axis=1)

    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs

    # new_shape = tf.convert_to_tensor([tf.shape(bbox)[0], tf.shape(bbox)[
    #                                  1], FLAGS.num_classes, tf.shape(bbox)[-1]])
    # bbox = tf.expand_dims(bbox, -2)
    # boxes = tf.broadcast_to(bbox, new_shape)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(

        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold

    )

    return boxes, scores, classes, valid_detections

# implement main YOLOv3 network


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input(shape=(size, size, channels), name='input')
    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    # large-scale-detection-featurevector {eg. 13x13 grid}
    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    # medium-scale-detection-featurevector {eg. 26x26 grid}
    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    # small-scale-detection-featurevector {eg. 52x52 grid}
    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    # if training is true return outputs for loss claculation else
    # return detection results ie. yolo_nms
    if training:
        return Model(inputs, [output_0, output_1, output_2], name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')

# Darknet for Yolov3Tiny :


def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    # FeatureVector1 -> for medium scale detection
    x = x_8 = DarknetConv(x, 256, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    # FeatureVector2 -> for large scale detection
    x = DarknetConv(x, 1024, 3)
    return Model(inputs, (x_8, x), name=name)

# YoloConvTiny for YOLOv3Tiny


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3_tiny')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')


def YoloLoss(anchors, classes, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # y_pred:(batch,grid,grid,anchors,(tx,ty,tw,th,obj,one-hot-classes))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        # pred_xywh == tx,ty,tw,th where sigmoid applied to tx,ty to get offset
        # since pred_txty can be any value but we want offsets so we used
        # called yolo_boxes wich returns pred_box==(x1,y1,x2,y2) for masking,
        # objctness,class score as well as original txtytwth to calculate loss
        pred_txty = pred_xywh[..., 0:2]
        pred_twth = pred_xywh[..., 2:4]

        # Transform all true outputs
        # y_true:(batch_size,grid,grid,anchors,(x1, y1, x2, y2, obj, class)
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4])/2
        true_wh = (true_box[..., 2:4] - true_box[..., 0:2])

        # Inverse transformation on the true_box to get tx,ty,tw,th
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        # ->two arrays (grid_size,grid_size)--> X,Y
        grid = tf.stack(grid, axis=-1)
        # (13,13,2)--eg.array([[[0,0],[1,0],[2,0]...],[0,1],..]]])
        grid = tf.expand_dims(grid, axis=2)  # (13,13,1,2)

        # tx = [bx * grid_size] - Cx; Cx = grid_x
        # ty = [by * grid_size] - Cy; Cy = grid_y
        true_txty = true_xy * \
            tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)

        # tw = ln(bw/Pw)
        # th = ln(bh/Ph)
        true_twth = tf.math.log(true_wh/anchors)
        true_twth = tf.where(tf.math.logical_or(tf.math.is_inf(true_twth), tf.math.is_nan(
            true_twth)), tf.zeros_like(true_twth), tf.math.abs(true_twth))  # <---------------{{{{

        # obj mask: by masking obj-loss we can teach network to detect region of interest
        obj_mask = tf.squeeze(true_obj, -1)

        # noobj_loss: since we don't want the network to cheat by proposing object
        # everywhere. Hence we need noobj_loss to penalize those false positive
        # proposals.
        # We get false positive by masking predictions with (1-obj_mask)
        # The 'ignore_mask' is used to make sure we only penalize when the current
        # box doesn't overlap much with ground truth box.
        best_iou = tf.map_fn(lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
            x[1], tf.cast(x[2], tf.bool))), axis=-1), (pred_box, true_box, obj_mask), tf.float32)

        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # λ_coord: its the weight that joseph redmon introduces in Yolov1
        # to put more emphasis on localization instead of classification.
        # In his paper he chose it to be 5 but we will chose it in a way
        # as to give more emphasis on small boxes than bigger ones:
        lambda_coord = 2 - true_wh[..., 0] * true_wh[..., 1]

        # since there are way too many noobj than obj in our fround_truth, we need
        # λ_noobj = 1
        lambda_noobj = 1

        # Calculate Loss

        # factor for confidence focal loss
        conf_focal = tf.pow(obj_mask-tf.squeeze(pred_obj, -1), 2)

        xy_loss = obj_mask * \
            tf.reduce_sum(tf.square(true_txty-pred_txty), axis=-1)

        wh_loss = obj_mask * \
            tf.reduce_sum(tf.square(true_twth-pred_twth), axis=-1)

        # factor for confidence focal loss
        conf_focal = tf.pow(obj_mask-tf.squeeze(pred_obj, -1), 2)

        obj_loss = conf_focal*binary_crossentropy(
            true_obj, pred_obj, from_logits=False)*obj_mask

        noobj_loss = lambda_noobj*(1-obj_mask)*ignore_mask*obj_loss

        # focal cant be implemented with sparse
        # use binary cross entropy instead
        class_focal = tf.pow(obj_mask-tf.squeeze(pred_class, -1), 2)

        class_loss = class_focal * obj_mask * \
            sparse_categorical_crossentropy(
                true_class_idx, pred_class, from_logits=False)
        # since categorical cross entropy expects softmax activations

        # sum over (batch,grid_y,grid_x,anchors) => (batch,1)
        xy_loss = tf.reduce_mean(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_mean(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        noobj_loss = tf.reduce_sum(noobj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + noobj_loss + class_loss
    return yolo_loss
