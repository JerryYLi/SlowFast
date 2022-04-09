#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
import csv
import random
import numpy as np
from collections import defaultdict

from slowfast.utils.env import pathmgr

logger = logging.getLogger(__name__)

FPS = 30
TRAIN_PERSON_INDICES = range(1, 26)
NUM_CLASSES_VERB = 125
NUM_CLASSES_NOUN = 352


def sec_to_frame(sec):
    """Time index (in seconds) to frame index."""
    return int(np.round(float(sec) * FPS))


def frame_to_sec(frame):
    """Frame index to time index (in seconds)."""
    return int(np.round(float(frame) / FPS))


def time_to_sec(sec):
    """Parse time string. (e.g. "00:02:10.99")"""
    hour, minute, sec = sec.split(':')
    return 3600.0 * int(hour) + 60.0 * int(minute) + float(sec)


def load_image_lists(cfg, is_train, return_dict=False):
    """Load frame paths and annotations. """

    list_filenames = [
        os.path.join(cfg.EPIC.FRAME_LIST_DIR, filename)
        for filename in (
            cfg.EPIC.TRAIN_LISTS
            # if (is_train or cfg.GET_TRAIN_LFB)
            if is_train
            else cfg.EPIC.TEST_LISTS)
    ]

    image_paths = defaultdict(list)
    labels = defaultdict(list)

    video_name_to_idx = {}
    video_idx_to_name = {}
    for list_filename in list_filenames:
        with open(list_filename, 'r') as f:
            f.readline()
            for line in f:
                row = line.split()
                # original_vido_id video_id frame_id path labels
                assert len(row) == 5
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name[idx] = video_name

                if return_dict:
                    data_key = video_name
                else:
                    data_key = video_name_to_idx[video_name]

                image_paths[data_key].append(os.path.join(cfg.EPIC.FRAME_DIR, row[3]))

                frame_labels = row[-1].replace('\"', '')
                if frame_labels != "":
                    labels[data_key].append(map(int, frame_labels.split(',')))
                else:
                    labels[data_key].append([])

    if return_dict:
        image_paths = dict(image_paths)
        labels = dict(labels)
    else:
        image_paths = [image_paths[i] for i in range(len(image_paths))]
        labels = [labels[i] for i in range(len(labels))]
    return image_paths, labels, video_idx_to_name, video_name_to_idx


def get_sequence(start_frame, stop_frame, half_len,
                 sample_rate, num_frames, is_train):
    """Get a sequence of frames (for a clip) with appropriete padding."""
    if is_train:
        center_frame = random.randint(start_frame, stop_frame)
    else:
        center_frame = (stop_frame + start_frame) // 2
    # seq = range(center_frame - half_len, center_frame + half_len, sample_rate)
    seq = list(range(center_frame - half_len, center_frame + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1

    return seq, center_frame


def load_annotations(cfg, is_train):
    """Load EPIC-Kitchens annotations."""

    annotations = []

    verb_set = set()
    noun_set = set()

    filename = os.path.join(cfg.EPIC.ANNOTATION_DIR, cfg.EPIC.ANNOTATIONS)
    # filename = "EPIC_100_train.csv" if is_train else "EPIC_100_validation.csv"
    # filename = os.path.join(cfg.EPIC.ANNOTATION_DIR, filename)
    # with open(filename, 'rb') as f:
    with open(filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            person = row[1]

            if is_train:
                if int(person[1:]) not in TRAIN_PERSON_INDICES:
                    continue
            else:
                if int(person[1:]) in TRAIN_PERSON_INDICES:
                    continue

            #uid,participant_id,video_id,narration,start_timestamp,stop_timestamp,start_frame,stop_frame,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
            video_name = row[2]
            start_frame = sec_to_frame(time_to_sec(row[4]))
            stop_frame = sec_to_frame(time_to_sec(row[5]))
            verb = int(row[-5])
            noun = int(row[-3])

            assert verb < NUM_CLASSES_VERB, verb
            assert verb >= 0, verb
            assert noun < NUM_CLASSES_NOUN, noun
            assert noun >= 0, noun

            annotations.append(
                (person, video_name, start_frame, stop_frame, verb, noun))
            verb_set.add(verb)
            noun_set.add(noun)

    logger.info('See %d verbs and %d nouns in the dataset loaded.' % (
        len(verb_set), len(noun_set)))

    cur_label_set = verb_set if cfg.EPIC.CLASS_TYPE == 'verb' else noun_set
    if len(cur_label_set) != cfg.MODEL.NUM_CLASSES:
        logger.warn(
            '# classes seen (%d) != MODEL.NUM_CLASSES' % len(cur_label_set))
    # assert len(annotations) == (cfg.TRAIN.DATASET_SIZE if is_train
    #                             else cfg.TEST.DATASET_SIZE)
    return annotations


# def get_annotations_for_lfb_frames(cfg, image_paths):
#     """
#     Return the "annotations" that correspond to the frames/clips that will be
#     used to construct LFB. The sampling is done uniformly with frequency
#     controlled by EPIC.VERB_LFB_CLIPS_PER_SECOND.
#     """
#     annotations = []

#     sample_freq = FPS // cfg.EPIC.VERB_LFB_CLIPS_PER_SECOND

#     for video_name in image_paths.keys():
#         for img_path in image_paths[video_name]:

#             frame = filename_to_frame_id(img_path)
#             if frame % sample_freq == 0:
#                 annotations.append((video_name[:3], video_name, frame, frame, 0, 0))

#     return annotations


# def filename_to_frame_id(img_path):
#     return int(img_path[-10:-4])


# def sample_verb_lfb(cfg, center_idx, video_lfb):
#     """Sample verb LFB."""
#     window_size = cfg.LFB.WINDOW_SIZE
#     half_len = (window_size * FPS) // 2

#     lower = center_idx - half_len
#     upper = center_idx + half_len

#     out_lfb = []
#     for frame_idx in range(lower, upper + 1):
#         if frame_idx in video_lfb.keys():
#             if len(out_lfb) < window_size:
#                 out_lfb.append(video_lfb[frame_idx])

#     out_lfb = np.array(out_lfb)
#     if out_lfb.shape[0] < window_size:
#         new_out_lfb = np.zeros((window_size, cfg.LFB.LFB_DIM))
#         if out_lfb.shape[0] > 0:
#             new_out_lfb[:out_lfb.shape[0]] = out_lfb
#         out_lfb = new_out_lfb

#     return out_lfb.astype(np.float32)


# def is_empty_list(x):
#     return isinstance(x, (list,)) and len(x) == 0


# def sample_noun_lfb(cfg, center_idx, video_lfb):
#     """Sample noun LFB."""
#     max_num_feat_per_frame = cfg.EPIC.MAX_NUM_FEATS_PER_NOUN_LFB_FRAME
#     window_size = cfg.LFB.WINDOW_SIZE

#     secs = float(window_size) / (max_num_feat_per_frame
#                                  * cfg.EPIC.NOUN_LFB_FRAMES_PER_SECOND)
#     lower = int(center_idx - (secs / 2) * FPS)
#     upper = int(lower + secs * FPS)

#     out_lfb = []
#     num_feat = 0
#     for frame_idx in range(lower, upper + 1):
#         if frame_idx in video_lfb:
#             frame_lfb = video_lfb[frame_idx]
#             if not is_empty_list(frame_lfb):
#                 curr_num = min(max_num_feat_per_frame, frame_lfb.shape[0])
#                 num_feat += curr_num

#                 out_lfb.append(frame_lfb[:curr_num])
#                 if num_feat >= window_size:
#                     break

#     if len(out_lfb) == 0:
#         logger.warn('No LFB sampled (certer_idx: %d)' % center_idx)
#         return np.zeros((window_size, cfg.LFB.LFB_DIM))

#     out_lfb = np.vstack(out_lfb)[:window_size].astype(np.float32)

#     if random.random() < 0.001:
#         logger.info(out_lfb.shape)
#     if out_lfb.shape[0] < window_size:
#         new_out_lfb = np.zeros((window_size, cfg.LFB.LFB_DIM))
#         new_out_lfb[:out_lfb.shape[0]] = out_lfb
#         out_lfb = new_out_lfb

#     return out_lfb