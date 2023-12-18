#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:18:10 2017

@author: shahidm
"""

from os.path import realpath, join, abspath, dirname


# defaults
SCRIPT_PATH = dirname(realpath(__file__))

NETWORK_MASKS_DIR = abspath(join(SCRIPT_PATH, 'network_masks'))
DSI_EDDY_SHELL_BVAL = abspath(join(SCRIPT_PATH, 'DSI_eddy_shells.bval'))
SLSPEC_FILE = abspath(join(SCRIPT_PATH, 'mb_slice_order_93slices.txt'))

SEQ_PARAMS_FILE = abspath(join(SCRIPT_PATH, 'SeqParams_RhinelandStudy_iso15.txt'))
DIRS_RAW = abspath(join(SCRIPT_PATH, 'dirs_raw257_mix.txt'))
DIRS_JONES = abspath(join(SCRIPT_PATH, 'CS_dirs_jones_N112_dsi.txt'))

DSISORTB0_RECON_IDX = abspath(join(SCRIPT_PATH, 'DiffusionDSIsortb0_recon_idx.txt'))

ECHO_SPACING_MSEC = 0.680
ECHO_TRAIN_LENGTH = 98
PA_NUM = 4

