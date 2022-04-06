#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from fvcore.common.config import CfgNode

def add_epic_config(_C):
    # -----------------------------------------------------------------------------
    # EPIC-Kitchens Dataset options
    # -----------------------------------------------------------------------------
    _C.EPIC = CfgNode()

    # Directory path of frames.
    _C.EPIC.FRAME_DIR = "data/epic/data/frames/"

    # Directory path for files of frame lists.
    _C.EPIC.FRAME_LIST_DIR = (
        # "data/epic/data/frame_lists/"
        "data/epic/data/frame_lists_55/"
    )

    # Directory path for annotation files.
    _C.EPIC.ANNOTATION_DIR = (
        # "data/epic/data/annotations/"
        "data/epic/data/annotations_55/"
    )

    # Type of target class (verb or noun).
    _C.EPIC.ANNOTATIONS = "EPIC_train_action_labels.csv"

    # Filenames of training samples list files.
    _C.EPIC.TRAIN_LISTS = ["train.csv"]

    # Filenames of test samples list files.
    _C.EPIC.TEST_LISTS = ["val.csv"]

    # Type of target class (verb or noun).
    _C.EPIC.CLASS_TYPE = "noun"

    # Whether to do center crop instead of full-resolution input during test.
    _C.EPIC.TEST_CENTER_CROP = False

    # Whether to do horizontal flipping during test.
    _C.EPIC.TEST_FORCE_FLIP = False

    # Whether to use full test set for validation split.
    _C.EPIC.FULL_TEST_ON_VAL = False

    # # The name of the file to the ava label map.
    # _C.EPIC.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

    # # The name of the file to the ava exclusion.
    # _C.EPIC.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

    # # The name of the file to the ava groundtruth.
    # _C.EPIC.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"
