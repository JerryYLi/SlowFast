#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch

from . import epic_helper as epic_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Epic(torch.utils.data.Dataset):
    """
    EPIC-Kitchens Dataset
    """

    def __init__(self, cfg, mode, num_retries=10):
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Something-Something V2".format(mode)
        self.mode = mode
        self.cfg = cfg
        
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        self._num_retries = num_retries
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP

        self._load_data(cfg)

    def _load_data(self, cfg):
        """Load frame paths and annotations. """

        # Load frame paths.
        (self._image_paths,
         self._image_labels,
         self._video_idx_to_name,
         self._video_name_to_idx) = epic_helper.load_image_lists(
            cfg, is_train=(self.mode == "train"), return_dict=True)

        # Load annotations.
        # if self._lfb_infer_only:
        #     self._annotations = epic_helper.get_annotations_for_lfb_frames(
        #         cfg, image_paths=self._image_paths
        #     )
        #     logger.info(
        #         'Inferring LFB from %d clips in %d videos.' % (
        #             len(self._annotations), len(self._image_paths)))
        # else:
        self._annotations = epic_helper.load_annotations(
            cfg, is_train=(self.mode == "train")
        )

        self.print_summary()

    def print_summary(self):
        logger.info("=== EPIC Kitchens dataset summary ===")
        logger.info('Split: {}'.format(self.mode))
        # logger.info("Use LFB? {}".format(self._lfb_enabled))
        # logger.info('Spatial shift position: {}'.format(self._shift))
        logger.info('Number of videos: {}'.format(len(self._image_paths)))
        total_frames = sum(len(video_img_paths)
                           for video_img_paths in self._image_paths.values())
        logger.info('Number of frames: {}'.format(total_frames))
        logger.info('Number of annotations: {}'.format(len(self._annotations)))
    
    # def sample_lfb(self, video_name, center_idx):
    #     """Sample LFB. Note that for verbs, we use video-model based LFB, and
    #     for nouns, we use detector-based LFB, so the formats are slightly
    #     different. Thus we use different functions for verb LFB and noun LFB."""
    #     if self.cfg.EPIC.CLASS_TYPE == 'noun':
    #         return epic_helper.sample_noun_lfb(
    #             self.cfg, center_idx, self._lfb[self._video_name_to_idx[video_name]]
    #         )
    #     else:
    #         return epic_helper.sample_verb_lfb(
    #             self.cfg, center_idx, self._lfb[video_name]
    #         )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._image_paths)

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(idx, tuple):
            idx, short_cycle_idx = idx

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            # spatial_sample_index = (
            #     self._spatial_temporal_idx[index]
            #     % self.cfg.TEST.NUM_SPATIAL_CROPS
            # )
            assert self.cfg.TEST.NUM_SPATIAL_CROPS == 1, "Multi-crop testing not supported"
            spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        (person, video_name,
            start_frame, stop_frame, verb, noun) = self._annotations[idx]

        # Get the frame idxs for current clip.
        seq, center_idx = epic_helper.get_sequence(
            start_frame, 
            stop_frame, 
            self._seq_len // 2, 
            self._sample_rate,
            len(self._image_paths[video_name]), 
            (self.mode == "train")
        )

        label = verb if self.cfg.EPIC.CLASS_TYPE == 'verb' else noun
        # if self._lfb_enabled:
        #     lfb = self.sample_lfb(video_name, center_idx)

        # Load images of current clip.
        frames = torch.as_tensor(
            utils.retry_load_images(
                [self._image_paths[video_name][frame] for frame in seq], 
                self._num_retries
            )
        )

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )

        imgs = utils.pack_pathway_output(self.cfg, imgs)
        return imgs, label, idx, {}