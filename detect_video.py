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
flags.DEFINE_string('output', 'output.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'MP4V',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

# track_iou parameters
flags.DEFINE_float('sigma_l', 0.2, 'sigma_l for low iou')
flags.DEFINE_float('sigma_h', 0.9, 'sigma high')
flags.DEFINE_float('sigma_iou', 0.8, 'IOU for tracking')
flags.DEFINE_integer('t_min', 4, 'min number of frames to track')


def proj3DWorld(p, n, delta):
    # remember to import numpy
    delta = np.array([delta])
    rho = np.array(np.concatenate([n, delta], axis=0))
    P = -(delta*p)/np.dot((np.array(np.concatenate([p, [0]], axis=-1))), rho)
    return P


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

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        output = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    info_moving = 1
    frame_num = 1
    det_frames_stack = []
    W = width
    H = height
    u = np.array([(444-W/2), (-(26-H/2)), 1])
    v = np.array([(530-W/2), (-(22-H/2)), 1])

    f = np.sqrt(np.dot(u, v))
    u_vec = np.array([(444-W/2), (-(26-H/2)), f])
    v_vec = np.array([(530-W/2), (-(22-H/2)), f])

    w_vec = np.cross(u_vec, v_vec)
    n_vec = w_vec/np.linalg.norm(w_vec)

    delta = 1
    # ---------------------------------------

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        info_moving += 1
        frame_num += 1

        # (H, W) = img.shape[0:2]

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        # img = cv2.rectangle(img, (0, 1), (65, 10), (255, 255, 255), -1)
        # img = cv2.putText(img, "Num: {}".format(nums[0]), (0, 10),
        #                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        # img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
        #                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # convert boxes,scores,classes into a list of [boxes,scores,classes]--- tensors
        detections = tf.concat([tf.reshape(boxes, (tf.shape(boxes)[1], tf.shape(boxes)[-1])), tf.reshape(
            scores, (tf.shape(scores)[1], 1)), tf.reshape(classes, (tf.shape(classes)[1], 1))], axis=1)
        # only top valid detections not all hundred
        detections = detections[0:nums[0]].numpy()

        det_list = []
        for i in range(nums[0]):
            det_list.append(detections[i])

        det_frames_stack.append(det_list)

        tracks = track_iou(det_frames_stack, FLAGS.sigma_l,
                           FLAGS.sigma_h, FLAGS.sigma_iou, FLAGS.t_min)
        if frame_num > FLAGS.t_min:
            det_frames_stack.pop(0)

        id_ = 1
        tracks_out = []

        # -----------------------------------------

        for track in tracks:
            outs = []
            for i, bbox in enumerate(track['bboxes']):
                outs += [(((bbox[0] + (bbox[2]-bbox[0])/2)*W).astype(np.int32), ((bbox[1]+(bbox[3]-bbox[1])/2)*H).astype(np.int32), ((bbox[2]-bbox[0])*W).astype(np.int32), ((bbox[3]-bbox[1])*H).astype(np.int32),
                          (track['start_frame'] + i), (id_))]
            id_ += 1
            tracks_out.append(outs)
        net_speed = 0
        for track_out in tracks_out:
            mean_speed = 0
            # speed determination for each track with orthographic projection
            # u_x,y = 444,26 # manually chosen
            # v_x,y = 530,22 # manually chosen
            # with respect to principal point c = center of image:
            # transformation = x->x-a, y->-(y-b) since y is calculated from top left in cv2; a = W/2, H/2
            # therefore, u_x,y = 444-W/2,-(26-H/2)
            #            v_x,y = 530-W/2,-(22-H/2)
            # f = √(u.v)
            # u_vector = [u_x,u_y,f]^T
            # v_vector = [v_x,v_y,f]^T
            # w_vector = u_vector X v_vector
            # n_vector = w_vector/|w_vector|
            # rho_vecor = [n^T,delta] # road plane
            # p1_vector = [track_out[1][0]#p1_x,track_out[1][1]#p1_y, f]^T
            # 3D world coord:
            # P1_vec = [-delta/([(p1_vector)^T,0].rho_vector)]*p1_vector----(1)
            # similarly for point 2
            # lambda_j(for jth scale) = l_t_j/||Front-Rear||
            # Front by equation(1), similarly Rear, lt = say average length of vehicle(5 metres)
            for i in range(len(track_out)-2):
                p1 = np.array([track_out[i][0], track_out[i][1], f])
                P1_vec = proj3DWorld(p1, n_vec, delta)

                p2 = np.array([track_out[i+2][0], track_out[i+2][1], f])
                P2_vec = proj3DWorld(p2, n_vec, delta)

                f1 = np.array(
                    [(track_out[i][0]), (track_out[i][1]+track_out[i][3]/2), f])
                r1 = np.array(
                    [(track_out[i][0]+track_out[i][2]/3), (track_out[i][1]-track_out[i][3]/3), f])

                # f2 = np.array([(track_out[-2][0]), (track_out[-2][1]+track_out[-2][3]/2),f])
                # r2 = np.array([(track_out[-2][0]+track_out[-2][2]/4), (track_out[-2][1]),f])

                F1 = proj3DWorld(f1, n_vec, delta)
                R1 = proj3DWorld(r1, n_vec, delta)

                mean_vehicle_length = 5.47  # in meters

                scale = mean_vehicle_length / np.linalg.norm((F1-R1))

                dist = scale * np.linalg.norm((P1_vec - P2_vec))
                # logging.info(f'dist:{dist}')

                dtime = 2/fps
                # logging.info(f'delta-t{dtime}')

                speed = (dist/dtime) * 3.6  # mps to kmph
            #     mean_speed += speed
            # mean_speed = mean_speed/len(track_out)

            net_speed += speed
            for i in range(len(track_out)):
                img = cv2.rectangle(
                    img, (track_out[-1][0], track_out[-1][1]), (track_out[-1][0]+130, track_out[-1][1]-10), (0, 255, 0), -1)
                img = cv2.putText(img, "speed:{:.1f}km/h".format(speed), (track_out[-1][0], track_out[-1][1]),
                                  cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            if track_out[-1][1] > track_out[-2][1]:
                img = cv2.rectangle(
                    img, (track_out[-1][0], track_out[-1][1]), (track_out[-1][0]+70, track_out[-1][1]-10), (0, 0, 255), -1)
                img = cv2.putText(img, "WRONG!", (track_out[-1][0], track_out[-1][1]),
                                  cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        traffic_mean_speed = net_speed/nums[0]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.rectangle(img, (0, 1), (65, 10), (0, 0, 0), -1)
        img = cv2.putText(img, "#total: {}".format(nums[0]), (0, 10),
                          cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        # img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
        #                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # traffic moving or not:
        if traffic_mean_speed < 5 and frame_num > FLAGS.t_min:
            logging.info('traffic_jam!')
            img = cv2.rectangle(img, (0, 70), (65, 58), (255, 255, 255), -1)
            img = cv2.putText(img, "TRAF-JAM!", (0, 80),
                              cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        else:
            # logging.info('moving...!')
            img = cv2.rectangle(img, (0, 70), (65, 58), (255, 255, 255), -1)
            status = "Moving..."
            img = cv2.putText(img, status, (0, 80),
                              cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # traffic dense/normal/congested/
        if nums[0] < 20:
            logging.info('Normal!')
            img = cv2.rectangle(img, (0, 70), (65, 10), (255, 255, 255), -1)
            img = cv2.putText(img, "Normal!", (0, 80),
                              cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        elif nums[0] < 30 and nums[0] > 20:
            logging.info('dense!')
            img = cv2.rectangle(img, (0, 70), (65, 10), (255, 255, 255), -1)
            img = cv2.putText(img, "dense!", (0, 80),
                              cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        elif nums[0] > 30:
            logging.info('very_Dense/congested!')
            img = cv2.rectangle(img, (0, 70), (65, 10), (255, 255, 255), -1)
            img = cv2.putText(img, "Very Dense / Congested!", (0, 80),
                              cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        tracks_out.clear()
        tracks.clear()

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
