import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from models import (
    YoloV3, YoloV3Tiny
)
from dataset import transform_images
from utils import draw_outputs
from iou_tracker import track_iou
import numpy as np

flags.DEFINE_string('classes', './coco.names', 'path to classes file')
flags.DEFINE_string('weights', './yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', 'output', 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

# track_iou parameters
flags.DEFINE_float('sigma_l', 0.2, 'sigma_l for low iou')
flags.DEFINE_float('sigma_h', 0.9, 'sigma high')
flags.DEFINE_float('sigma_iou', 0.8, 'IOU for tracking')
flags.DEFINE_integer('t_min', 4, 'min number of frames to track')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        output = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 1
    det_frames_stack = []

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        frame_num += 1

        (W, H) = img.shape[0:2]

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.rectangle(img, (0, 1), (65, 10), (255, 255, 255), -1)
        img = cv2.putText(img, "Num: {}".format(nums[0]), (0, 10),
                          cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # convert boxes,scores,classes into a list of [boxes,scores,classes]--- tensors
        detections = tf.concat([tf.reshape(boxes, (tf.shape(boxes)[1], tf.shape(boxes)[-1])), tf.reshape(
            scores, (tf.shape(scores)[1], 1)), tf.reshape(classes, (tf.shape(classes)[1], 1))], axis=1)
        # only top valid detections not all hundred
        detections = detections[0:nums[0]].numpy()

        det_list = []
        for i in range(nums[0]):
            det_list.append(detections[i])

        det_frames_stack.append(det_list)

        if frame_num > FLAGS.t_min:
            tracks = track_iou(det_frames_stack)
            det_frames_stack.pop(0)

        id_ = 1
        out = []
        for track in tracks:
            for i, bbox in enumerate(track['bboxes']):
                out += [((bbox[0]*W).astype(np.int8), (bbox[1]*H).astype(np.int8),
                         (track['start_frame'] + i), (id_))]
            id_ += 1
        for i in range(len(out)-1):
            img = cv2.rectangle(img, (0, 1), (65, 10), (255, 255, 255), -1)
            img = cv2.putText(img, "Num: {}".format(nums[0]), (0, 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.rectangle(img, (0, 1), (65, 10), (255, 255, 255), -1)
        img = cv2.putText(img, "Num: {}".format(nums[0]), (0, 10),
                          cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # -------------------------------end------------- show output
        if FLAGS.output:
            output.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
