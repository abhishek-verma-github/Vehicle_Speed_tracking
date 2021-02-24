import cv2
import tensorflow as tf
import numpy as np
from time import time
from models import broadcast_iou
from absl import logging


def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    IOU based tracker.
    --->"High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora"
    args:
         detections (list): list of detections per frame,
         sigma_l (float): low detection threshold,
         sigma_h (float): high detection threshold,
         sigma_iou (float): IOU threshold,
         t_min (float): minimum track length in frames,
    Returns:
        list: list of tracks.
    """

    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det[4] >= sigma_l]
        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get detection with highest iou
                best_match = max(dets, key=lambda x: broadcast_iou(
                    track['bboxes'][-1], x[0:4]))
                if broadcast_iou(track['bboxes'][-1], best_match[0:4]) >= sigma_iou:
                    track['bboxes'].append(best_match[0:4])
                    track['max_score'] = max(track['max_score'], best_match[4])
                    updated_tracks.append(track)

                    # remove from best matching detections from detection
                    # dets.remove(best_match.any())

            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when conditions are met
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'bboxes': [det[0:4]], 'max_score': det[4],
                       'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished
