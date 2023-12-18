from __future__ import division

import nipype.pipeline.engine as pe
from nipype import SelectFiles
import nipype.interfaces.utility as util

from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import (ApplyMask,UnaryMaths)

from nipype import IdentityInterface, DataSink
from .mdt_interface import MDTGenProtocol, MDTFitModel

from .dtiutil import (b_threshold,clip_max)

from .configoptions import SEQ_PARAMS_FILE
import os

from .fsl_tensor_decomp import FSLMathsCommand


def get_subs(subj_ids):
    subs=[('_subject_ids_%s'%subj_ids,''),
          ('_subject_ids_%s/_model_name_'%subj_ids,''),
          ('_MDT_Tensor/',''),('_MDT_NODDI/',''),
          ('FSL_DTIFIT_L1','FSL_DTIFIT_AD'),
          ('_MDT_Kurtosis','')]
    return subs

def get_file_path(mdt_out_dir):
    import os
    #additional S0 present in the path and path is a list
    if isinstance(mdt_out_dir, list):
        mdt_out_dir = mdt_out_dir[0]

    sigma_file=os.path.abspath(os.path.join(mdt_out_dir,'S0/OffsetGaussian.sigma.nii.gz'))
    return sigma_file


def get_noddi_files(mdt_out_dir):
    import os

    if isinstance(mdt_out_dir,list):
        mdt_out_dir=mdt_out_dir[0]
    used_mask = os.path.abspath(os.path.join(mdt_out_dir,'NODDI/UsedMask.nii.gz'))
    ndi_file  = os.path.abspath(os.path.join(mdt_out_dir,'NODDI/NDI.nii.gz'))

    return used_mask, ndi_file

def get_kurt_rk_file(mdt_out_dir):
    import os

    if isinstance(mdt_out_dir,list):
        mdt_out_dir=mdt_out_dir[0]
    rk_file  = os.path.abspath(os.path.join(mdt_out_dir,'Kurtosis/KurtosisTensor.RK.nii.gz'))

    return rk_file


def get_op_string(out_stat):
    return "-mul %f" % out_stat[1]

def get_op_file(l3_file):
    return [l3_file]


def create_dsi_modeling(input_dir, work_dir, outputdir, subject_ids,mdtproc,
                        name='dsi_modeling'):

    dwiwf = pe.Workflow(name=name)
    dwiwf.base_dir = work_dir

    inputnode = pe.Node(interface=IdentityInterface(
            fields=['subject_ids', 'outputdir']
            ),name='inputnode')

    inputnode.iterables = [('subject_ids', subject_ids)]
    inputnode.inputs.subject_ids = subject_ids
    inputnode.inputs.outputdir = outputdir

    #templates to select relevant files
    templates = {"bvals":    "{subject_id}/CSRecon/dsi.bvals",
                 "bvecs":    "{subject_id}/CSRecon/dsi.bvecs",
                 "dsi"          :"{subject_id}/CSRecon/dsi.nii.gz",
                 "eddycorrected":"{subject_id}/Eddy/eddy_corrected.nii.gz",
                 "eddycorrmask" :"{subject_id}/Eddy/eddy_corrected_b0s_mean_brain_mask_dil.nii.gz",
                 "bvals_sortb0" :"{subject_id}/Eddy/bvals_sortb0.bval",
                 "bvecs_sortb0" :"{subject_id}/Eddy/bvecs_sortb0.bvec"
                 }


    fileselector = pe.Node(SelectFiles(templates), name='fileselect')
    fileselector.inputs.base_directory = input_dir

    #%%#
    #1 b-threshold data
    bv_threshold = pe.Node(interface=util.Function(
            input_names=['bvals_file', 'bvecs_file','dsi_file'],
            output_names=['reduced_bvals', 'reduced_bvecs','reduced_dsi'],
            function=b_threshold), name='bv_threshold')


    #%%#
    #2 dtifit
    dtifit = pe.Node(interface=fsl.DTIFit(),name='fsldtifit')
    dtifit.inputs.base_name='FSL_DTIFIT'
    dtifit.inputs.args='--wls'

    #2.1 get RD
    #fslmaths FSL_dtifit_L2.nii.gz -add FSL_dtifit_L3.nii.gz -div 2 FSL_dtifit_RD.nii.gz
    dtifit_rd = pe.Node(interface=fsl.MultiImageMaths(),name='dtifit_rd')
    dtifit_rd.inputs.out_file='FSL_DTIFIT_RD.nii.gz'
    dtifit_rd.inputs.op_string = "-add %s -div 2"

    #%%#
    #3.1 invert mask
    binv_mask = pe.Node(interface=UnaryMaths(),name='binv_mask')
    binv_mask.inputs.operation='binv'


    #%%#
    #3.2 mdt-gen-protocol
    mdt_genprot_us = pe.Node(interface=MDTGenProtocol(), name='mdt_genprotocol_us')
    mdt_genprot_us.inputs.prtcl_name='dsi_cs.prtcl'
    mdt_genprot_us.inputs.seq_params_file=SEQ_PARAMS_FILE

    #%%#
    #3.3 mdt-fitmodel us
    mdt_fitmodel_us = pe.Node(interface=MDTFitModel(), name='mdt_modelfit_us')

    mdt_fitmodel_us.inputs.MDT='S0'
    mdt_fitmodel_us.n_procs=mdtproc
    mdt_fitmodel_us.inputs.use_gpu=True
    mdt_fitmodel_us.inputs.out_dir=""
    mdt_fitmodel_us.inputs.dev_ind=os.environ[
            'CUDA_VISIBLE_DEVICE'
            ] if 'CUDA_VISIBLE_DEVICE' in os.environ else "0"

    #%%#
    #3.4 get path to S0/OffsetGaussian.sigma file
    get_sigma_path=pe.Node(interface=util.Function(input_names=['mdt_out_dir'],
                                             output_names=['sigma_file'],
                                             function=get_file_path),name='get_sigma_path')

    #3.5 get intensity stats
    intensity_stats = pe.Node(interface=fsl.ImageStats(), name='intensity_stats')
    intensity_stats.inputs.op_string='-R'

    #3.6 estimate noise std
    noise_std = pe.Node(interface=fsl.ImageMaths(),name='noise_std')


    #%%#
    #4.1 mdt model fit
    mdt_genprot = pe.Node(interface=MDTGenProtocol(), name='mdt_genprotocol')
    mdt_genprot.inputs.prtcl_name='dsi.prtcl'
    mdt_genprot.inputs.seq_params_file=SEQ_PARAMS_FILE

    #4.2
    mdt_fitmodel_noddi = pe.Node(interface=MDTFitModel(), name='mdt_modelfit_noddi')
    mdt_fitmodel_noddi.inputs.MDT='NODDI'
    mdt_fitmodel_noddi.n_procs=mdtproc
    mdt_fitmodel_noddi.inputs.use_gpu=True
    mdt_fitmodel_noddi.inputs.out_dir=""
    mdt_fitmodel_noddi.inputs.dev_ind=os.environ[
            'CUDA_VISIBLE_DEVICE'
            ] if 'CUDA_VISIBLE_DEVICE' in os.environ else "0"


    mdt_fitmodel_kurt = pe.Node(interface=MDTFitModel(), name='mdt_modelfit_kurt')
    mdt_fitmodel_kurt.inputs.MDT='Kurtosis'
    mdt_fitmodel_kurt.n_procs=mdtproc
    mdt_fitmodel_kurt.inputs.use_gpu=True
    mdt_fitmodel_kurt.inputs.out_dir=""
    mdt_fitmodel_kurt.inputs.dev_ind=os.environ[
            'CUDA_VISIBLE_DEVICE'
            ] if 'CUDA_VISIBLE_DEVICE' in os.environ else "0"



    noddi_files = pe.Node(interface=util.Function(input_names=['mdt_out_dir'],
                                                  output_names=['used_mask','ndi_file'],
                                                  function=get_noddi_files),name='get_nodi_files')

    kurt_rk_file = pe.Node(interface=util.Function(input_names=['mdt_out_dir'],
                                                  output_names=['rk_file'],
                                                  function=get_kurt_rk_file),name='get_kurt_rk_file')

    #%%#
    #volume fraction vf in NODDI
    vol_frac = pe.Node(interface=fsl.MultiImageMaths(),name='vol_frac')
    vol_frac.inputs.out_file ='vf_ec.nii.gz'
    vol_frac.inputs.op_string='-sub %s'

    #clipped RK file
    rk_clip = pe.Node(interface=util.Function(
        input_names=['in_file'],
        output_names=['out_file'],
        function=clip_max), name='rk_clip')

    #%% collect outputs
    datasink = pe.Node(interface=DataSink(), name='datasinker')
    datasink.inputs.parameterization=True

    #subsitute _subj_ids_{ID} from outpaths
    subsgen = pe.Node(interface=util.Function(
            input_names=['subj_ids'],
            output_names=['substitutions'],
            function=get_subs), name='subsgen')


    # %% workflow connections
    dwiwf.connect(inputnode        , 'subject_ids',    subsgen,  'subj_ids')
    dwiwf.connect(subsgen          , 'substitutions',  datasink, 'substitutions')

    #step 1 b-threshold
    dwiwf.connect(inputnode        , 'subject_ids',  fileselector,'subject_id')
    dwiwf.connect(fileselector     , 'bvals',        bv_threshold, 'bvals_file')
    dwiwf.connect(fileselector     , 'bvecs',        bv_threshold, 'bvecs_file')
    dwiwf.connect(fileselector     , 'dsi',          bv_threshold, 'dsi_file')


    #step 2 dtifit
    dwiwf.connect(fileselector  , 'eddycorrmask' , dtifit, 'mask')
    dwiwf.connect(bv_threshold   , 'reduced_bvals', dtifit, 'bvals')
    dwiwf.connect(bv_threshold   , 'reduced_bvecs', dtifit, 'bvecs')
    dwiwf.connect(bv_threshold   , 'reduced_dsi'  , dtifit, 'dwi')

    #2.1
    dwiwf.connect(dtifit,  'L2',        dtifit_rd,        'in_file')
    dwiwf.connect(dtifit, ('L3',get_op_file), dtifit_rd, 'operand_files')

    #step 3.1
    dwiwf.connect(fileselector  , 'eddycorrmask' , binv_mask, 'in_file')

    #step 3.2
    dwiwf.connect(fileselector  , 'bvecs_sortb0' , mdt_genprot_us,'bvecs')
    dwiwf.connect(fileselector  , 'bvals_sortb0' , mdt_genprot_us,'bvals')


    #step 3.3
    dwiwf.connect(mdt_genprot_us, 'out_prtcl'    , mdt_fitmodel_us,'prtcl_file')
    dwiwf.connect(binv_mask     , 'out_file'     , mdt_fitmodel_us,'brain_mask')
    dwiwf.connect(fileselector  , 'eddycorrected', mdt_fitmodel_us,'data_file')

    #step 3.4
    dwiwf.connect(mdt_fitmodel_us, 'outputfolder', get_sigma_path, 'mdt_out_dir')

    #step 3.5
    dwiwf.connect(get_sigma_path, 'sigma_file', intensity_stats,'in_file')

    #step 3.6
    dwiwf.connect(fileselector,   'eddycorrmask',  noise_std,   'in_file')
    dwiwf.connect(intensity_stats, ('out_stat',get_op_string), noise_std,   'op_string')

    #step 4
    dwiwf.connect(fileselector  , 'bvecs',      mdt_genprot,'bvecs')
    dwiwf.connect(fileselector  , 'bvals',      mdt_genprot,'bvals')

    dwiwf.connect(mdt_genprot   ,'out_prtcl',   mdt_fitmodel_noddi, 'prtcl_file')
    dwiwf.connect(fileselector  ,'dsi',         mdt_fitmodel_noddi, 'data_file')
    dwiwf.connect(fileselector  ,'eddycorrmask',mdt_fitmodel_noddi, 'brain_mask')
    dwiwf.connect(noise_std     ,'out_file',    mdt_fitmodel_noddi, 'noise_std')

    dwiwf.connect(mdt_genprot   ,'out_prtcl',   mdt_fitmodel_kurt, 'prtcl_file')
    dwiwf.connect(fileselector  ,'dsi',         mdt_fitmodel_kurt, 'data_file')
    dwiwf.connect(fileselector  ,'eddycorrmask',mdt_fitmodel_kurt, 'brain_mask')
    dwiwf.connect(noise_std     ,'out_file',    mdt_fitmodel_kurt, 'noise_std')

    dwiwf.connect(mdt_fitmodel_noddi  ,'outputfolder', noddi_files,'mdt_out_dir')
    dwiwf.connect(noddi_files   ,'used_mask'   , vol_frac,   'in_file')
    dwiwf.connect(noddi_files   ,('ndi_file',get_op_file), vol_frac,'operand_files')

    dwiwf.connect(mdt_fitmodel_kurt  ,'outputfolder', kurt_rk_file,'mdt_out_dir')
    dwiwf.connect(kurt_rk_file   ,'rk_file'   , rk_clip,   'in_file')


    #outputs
    dwiwf.connect(inputnode    , 'subject_ids',     datasink, 'container')
    dwiwf.connect(inputnode    , 'outputdir',       datasink, 'base_directory')


    dwiwf.connect(dtifit    , 'V1',          datasink, 'FSL.@V1')
    dwiwf.connect(dtifit    , 'V2',          datasink, 'FSL.@V2')
    dwiwf.connect(dtifit    , 'V3',          datasink, 'FSL.@V3')
    dwiwf.connect(dtifit    , 'L1',          datasink, 'FSL.@L1')
    dwiwf.connect(dtifit    , 'L2',          datasink, 'FSL.@L2')
    dwiwf.connect(dtifit    , 'L3',          datasink, 'FSL.@L3')
    dwiwf.connect(dtifit    , 'MD',          datasink, 'FSL.@MD')
    dwiwf.connect(dtifit    , 'FA',          datasink, 'FSL.@FA')
    dwiwf.connect(dtifit    , 'MO',          datasink, 'FSL.@MO')
    dwiwf.connect(dtifit    , 'S0',          datasink, 'FSL.@S0')
    dwiwf.connect(dtifit    , 'tensor',      datasink, 'FSL.@tensor')
    dwiwf.connect(dtifit_rd , 'out_file',    datasink, 'FSL.@RD')

    dwiwf.connect(mdt_fitmodel_kurt,
                  'outputfolder',       datasink,  'MDT.@outputfolder_kurt')
    dwiwf.connect(mdt_fitmodel_noddi,
                  'outputfolder',       datasink,  'MDT.@outputfolder_noddi')


    dwiwf.connect(vol_frac , 'out_file',    datasink, 'MDT.NODDI.@vf')
    dwiwf.connect(rk_clip  , 'out_file',    datasink, 'MDT.Kurtosis.@dk_rk_clip')

    return dwiwf
