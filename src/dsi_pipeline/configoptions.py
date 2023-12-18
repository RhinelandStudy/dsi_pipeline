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


atlasMNI_T1       =abspath(join(SCRIPT_PATH, 'FSL_labels/MNI152_T1_1mm.nii.gz'))
atlasMNI_FA       =abspath(join(SCRIPT_PATH, 'FSL_labels/JHU-ICBM-FA-1mm.nii.gz'))
labelsROIsMNI     =abspath(join(SCRIPT_PATH, 'FSL_labels/JHU-ICBM-labels-1mm.nii.gz'))
labelsROIsMNI_RL  =abspath(join(SCRIPT_PATH, 'FSL_labels/JHU-ICBM-labels-1mm-RLcombined.nii.gz'))
labelsTractsMNI   =abspath(join(SCRIPT_PATH, 'FSL_labels/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz'))
skeletMNI         =abspath(join(SCRIPT_PATH, 'FSL_labels/JHU-ICBM-FA-skeleton-1mm.nii.gz'))
skeletMNI_PSMD    =abspath(join(SCRIPT_PATH, 'FSL_labels/PSMDS_skeleton_mask.nii.gz'))
labelsROIsHistMNI =abspath(join(SCRIPT_PATH, 'FSL_labels/Juelich-maxprob-thr25-1mm.nii.gz'))
configParams_s1   =abspath(join(SCRIPT_PATH, 'FSL_labels/optPar/oxford_s1.cnf'))
configParams_s2   =abspath(join(SCRIPT_PATH, 'FSL_labels/optPar/oxford_s2.cnf'))
configParams_s3   =abspath(join(SCRIPT_PATH, 'FSL_labels/optPar/oxford_s3.cnf'))
