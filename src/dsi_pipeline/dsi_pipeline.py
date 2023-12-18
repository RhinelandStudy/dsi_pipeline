"""
Copyright 2023 Population Health Sciences, German Center for Neurodegenerative Diseases (DZNE)
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0 
Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

"""

from __future__ import division

import nipype.pipeline.engine as pe
from nipype import SelectFiles
import nipype.interfaces.utility as util

from nipype.interfaces import fsl
from nipype.interfaces.fsl import ExtractROI
from nipype.interfaces.fsl import Eddy
from nipype.interfaces.fsl import FAST
from nipype.interfaces.fsl.maths import MeanImage
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.algorithms.misc import Gunzip

from nipype import IdentityInterface, DataSink
import glob

from .dtiutil import (create_encoding_params_file, extract_bvecs_bvals,
                      scanner2FOV, swap_bvecs, create_index_file,
                      unique_dwis_dsi,reorder_dwi,
                      reorder_bvals_bvecs,extractb0AP,GZip,crop_xy)


from .configoptions import (NETWORK_MASKS_DIR, DSI_EDDY_SHELL_BVAL,SLSPEC_FILE,
                            DIRS_RAW,DIRS_JONES,SEQ_PARAMS_FILE)

from .csrecon import CSRECON


def get_pve0(pvefiles):
    
    for pvefile in pvefiles:
        if 'pve_0' in pvefile:
            return pvefile


def get_subs(subj_ids):
    subs=[('_subject_ids_%s'%subj_ids,''),
          ('_subject_ids_%s/_model_name_'%subj_ids,''),
          ('_b0s_merged',''),('_csf',''),
          ('bvecs_swaped.bvecs_sortb0','bvecs_swaped_sortb0'),
          ('_MDT_TensorCascade/',''),('_MDT_NODDICascade/',''),
          ('_MDT_KurtosisCascade',''),
          ('eddy_corrected_b0s_merged_CSRecon_csf','dsi_recon515'),
          ('eddy_corrected_CSRecon','dsi_recon515')]
    return subs
   
    
    
def create_dsi_pipeline(scans_dir, work_dir, outputdir, subject_ids, 
                        csrecon_path, mcr_path, poolsize, 
                        name='dsi_pipeline'):
    
    dwiwf = pe.Workflow(name=name)
    dwiwf.base_dir = work_dir

    inputnode = pe.Node(interface=IdentityInterface(
            fields=['subject_ids','outputdir','network_masks']
            ),name='inputnode')
    
    
    inputnode.iterables = [('subject_ids', subject_ids)]
    inputnode.inputs.subject_ids = subject_ids
    inputnode.inputs.network_masks = [glob.glob(NETWORK_MASKS_DIR + '/*.gz')]
    inputnode.inputs.outputdir = outputdir

    templates = {"DSI": "{subject_id}/*DiffusionDSI.nii.gz",
                 "DSIR":"{subject_id}/*DiffusionDSI_r.nii.gz"}
                 
    fileselector = pe.Node(SelectFiles(templates), name='fileselect')
    fileselector.inputs.base_directory = scans_dir

    #%%#
    #1: crop x,y dim by delete 1st and last slices 
    cropxy_dsi = pe.Node(interface=util.Function(
            input_names=['in_file'], output_names=['out_file'],
            function=crop_xy), name='cropxy_dsi')

    cropxy_dsir = pe.Node(interface=util.Function(
            input_names=['in_file'], output_names=['out_file'],
            function=crop_xy), name='cropxy_dsir')
    #%%#
    #2: first step extract bvals bvecs from input nifti file
    extractbvecsbvals = pe.Node(interface=util.Function(
            input_names=['niftifile'], output_names=['bvals','bvecs'],
            function=extract_bvecs_bvals), name='extractbvecsbvals')
    
    
    #2.1:
    scanner2_FOV = pe.Node(interface=util.Function(
            input_names=['file_data', 'file_bvec','flag_x',
                         'flag_y','flag_z','prefix'],
                         output_names=['bvecs_file'],
                         function=scanner2FOV), name='scanner2FOV')
    
    scanner2_FOV.inputs.flag_x = True
    scanner2_FOV.inputs.flag_y = True
    scanner2_FOV.inputs.flag_z = False
    scanner2_FOV.inputs.prefix = ''
    
    #%%#
    #3: extract all b0 images from DSI 
    extractb0_AP = pe.Node(interface=util.Function(
            input_names=['in_dwi', 'in_bvals'],
            output_names=['b0_AP'],
            function=extractb0AP), name='extractb0_AP')
    
    #%%#
    #4: merge b0_AP and PA filenames
    list_b0AP_PA = pe.Node(util.Merge(2), name='list_b0AP_PA')
    
    #%%#
    #4.1: merge b0_AP and PA volumes
    fslmerge_b0AP_PA = pe.Node(interface=fsl.Merge(), name = 'fslmerge_b0AP_PA')
    fslmerge_b0AP_PA.inputs.dimension = 't'
    fslmerge_b0AP_PA.inputs.output_type = 'NIFTI_GZ'

    #%%#
    #5: create acqparams file
    create_acqparams = pe.Node(interface=util.Function(
            input_names=['in_bvals'],
            output_names=['acqparams_file'],
            function=create_encoding_params_file), name='create_acqparams')
    
    #%%#
    #6: create index file for topup
    create_index = pe.Node(interface=util.Function(
            input_names=['in_bvals'],
            output_names=['index_file'],
            function=create_index_file), name='create_index')
    
    #%%#
    #7: run topup
    topup = pe.Node(interface=fsl.TOPUP(), name = 'topup')
    topup.inputs.output_type   = 'NIFTI_GZ'
    topup.inputs.config        = 'b02b0_3.cnf'
   
 
    #%%#
    #8: apply bet on topup corrected file
    tc_bet = pe.Node(interface=fsl.BET(), name='tc_bet')
    tc_bet.inputs.frac=0.20
    tc_bet.inputs.mask = True
    
    #%%#
    #9: dilate tc_bet mask
    dilate_tc_bet_mask = pe.Node(interface=fsl.DilateImage(),
                                 name='dilate_tc_bet_mask')
    
    dilate_tc_bet_mask.inputs.operation = 'modal'
    dilate_tc_bet_mask.inputs.output_datatype = 'input'
   

    #%%#
    #10: run eddy_cuda
    eddy_cuda = pe.Node(interface=Eddy(),name='eddy_cuda')
    #the in_bval for running eddy on DSI is provided in a static file 
    #with b0 fixes on the shell,
    #and therefore we tell it is_shelled=True
    #otherwise eddy_cuda crashes with segfault coredump
    eddy_cuda.inputs.in_bval = DSI_EDDY_SHELL_BVAL
    #eddy_cuda.inputs.repol=True
    #eddy_cuda.inputs.cnr_maps=True
    #eddy_cuda.inputs.residuals=True
    eddy_cuda.inputs.slspec=SLSPEC_FILE
    eddy_cuda.inputs.use_cuda = True
    eddy_cuda.inputs.is_shelled = True



    #%%#
    #11: reorder eddy correct dwis based on b0s
    reorder_ec_dwi = pe.Node(interface=util.Function(
            input_names=['in_dwi','in_bval'],
            output_names=['b0_out_file','dwi_out_file'],
            function=reorder_dwi), name='reorder_ec_dwi')
    
    #%%#
    #12: create a mean image of the b0s only extracted from recorder_ec_dwi
    b0s_mean = pe.Node(interface=MeanImage(), name='b0s_mean')
    b0s_mean.inputs.dimension='T'
    
    
    #%%#
    #13: BET and create mask from the eddy_corrected b0s only mean image
    ec_bet = pe.Node(interface=fsl.BET(), name='ec_bet')
    ec_bet.inputs.frac=0.20
    ec_bet.inputs.mask = True
    ec_bet.inputs.no_output=True
    
    #%%#
    #14: dilate ec_bet mask
    dilate_ec_bet_mask = pe.Node(interface=fsl.DilateImage(),
                                 name='dilate_ec_bet_mask')
    
    dilate_ec_bet_mask.inputs.operation = 'modal'
    dilate_ec_bet_mask.inputs.output_datatype = 'input'
    
    
    #%%#
    #15: apply (the dilated) mask to the b0s mean image:
    #we can get brain and mask in 1 BET step, 
    #but here we apply the dilated mask to the mean b0s image.
    mask_b0s_mean = pe.Node(interface=ApplyMask(), name='mask_b0s_mean')
    

    #%%#
    #16: FAST segment
    fast_b0s_mean_masked = pe.Node(interface=FAST(), name='fast_b0s_mean_masked')
    fast_b0s_mean_masked.inputs.img_type = 2 # -t 2
    fast_b0s_mean_masked.inputs.number_classes = 4 # -n 4
    fast_b0s_mean_masked.out_basename = "_fast_"
    
    #%%#
    #17: combine filenames into a list
    list_b0s_dwis = pe.Node(util.Merge(2), name='list_b0s_dwis')
    
    
    #%%#
    #18: merge b0s and dwis
    fslmerge_b0sdwis = pe.Node(interface=fsl.Merge(), name='fslmerge_b0sdwis')
    fslmerge_b0sdwis.inputs.dimension = 't'
    fslmerge_b0sdwis.inputs.output_type = 'NIFTI_GZ'
    
 
    #%%#
    #19: reorder bvals bvecs
    sort_bvals_bvecs = pe.Node(interface=util.Function(
            input_names=['bvals','bvecs'],
            output_names=['bvals_sortb0','bvecs_sortb0'],
            function=reorder_bvals_bvecs), name='sort_bvals_bvecs')
    
    
    #%%#
    #20: sort eddy_rotated bvecs as sort_bvals_bvecs node
    sort_eddy_rotated_bvecs = pe.Node(interface=util.Function(
            input_names=['bvals','bvecs'],
            output_names=['bvals_sortb0','bvecs_sortb0'],
            function=reorder_bvals_bvecs), name='sort_eddy_rotated_bvecs')
    
    #%%#
    #21: unzip nii.gz to nii
    dwi_gunzipper    = pe.Node(interface=Gunzip(), name='dwi_gunzipper')
    mask_gunzipper   = pe.Node(interface=Gunzip(), name='mask_gunzipper')
    csfpve_gunzipper = pe.Node(interface=Gunzip(), name='csfpve_gunzipper')
    
    
    #%%#
    #22 pass on to CSRecon
    csrecon = pe.Node(interface=CSRECON(), name='csrecon')
    csrecon.n_procs = poolsize
    
    csrecon.inputs.csrecon_path = csrecon_path
    csrecon.inputs.mcr_path = mcr_path
    csrecon.inputs.poolsize = poolsize
    csrecon.inputs.seq_params = SEQ_PARAMS_FILE
    csrecon.inputs.dirs_raw = DIRS_RAW
    csrecon.inputs.dirs_jones = DIRS_JONES
    
    
    #%%#
    #22.1:
    swapbvecs_recon514b0s_norm = pe.Node(interface=util.Function(
            input_names=['bvecs', 'flag_x', 'flag_y', 'flag_z'],
            output_names=['swaped_bvecs'],
            function=swap_bvecs), name='swapbvecs_recon514b0s_norm')
    
    swapbvecs_recon514b0s_norm.inputs.flag_x=False
    swapbvecs_recon514b0s_norm.inputs.flag_y=True
    swapbvecs_recon514b0s_norm.inputs.flag_z=True


    #22.2:
    swapbvecs_recon515 = pe.Node(interface=util.Function(
            input_names=['bvecs', 'flag_x', 'flag_y', 'flag_z'],
            output_names=['swaped_bvecs'],
            function=swap_bvecs), name='swapbvecs_recon515')
    
    swapbvecs_recon515.inputs.flag_x=False
    swapbvecs_recon515.inputs.flag_y=True
    swapbvecs_recon515.inputs.flag_z=True
    
    
    #22.3:
    scanner2_FOV_recon514b0s_norm = pe.Node(interface=util.Function(
            input_names=['file_data', 'file_bvec','flag_x','flag_y','flag_z','prefix'],
            output_names=['bvecs_file'],
            function=scanner2FOV), name='scanner2_FOV_recon514b0s_norm') 
    
    scanner2_FOV_recon514b0s_norm.inputs.flag_x = True
    scanner2_FOV_recon514b0s_norm.inputs.flag_y = True
    scanner2_FOV_recon514b0s_norm.inputs.flag_z = False
    scanner2_FOV_recon514b0s_norm.inputs.prefix='recon514b0s_norm'


    #22.4:
    scanner2_FOV_recon515 = pe.Node(interface=util.Function(
            input_names=['file_data', 'file_bvec','flag_x','flag_y','flag_z','prefix'],
            output_names=['bvecs_file'],
            function=scanner2FOV), name='scanner2_FOV_recon515')
    
    scanner2_FOV_recon515.inputs.flag_x = True
    scanner2_FOV_recon515.inputs.flag_y = True
    scanner2_FOV_recon515.inputs.flag_z = False
    scanner2_FOV_recon515.inputs.prefix = 'recon515'
    
    
    #%%#
    #23: gzip outputs back    
    csrecon_gzipper = pe.Node(interface=GZip(), name='csrecon_gzip')
    
    csrecon_csf_gzipper = pe.Node(interface=GZip(), name='csrecon_csf_gzip')

    #%%#
    #24: apply mask to all b0 only images (8 b0s obtained from reorder_ec_dwi)
    ec_b0s_masked = pe.Node(interface=ApplyMask(), name='ec_b0s_masked') 


    #%%#
    #25: extract roi (fslroi) from csrecon output
    fslroi_csrecon = pe.Node(interface=ExtractROI(), name='fslroi_csrecon')
    fslroi_csrecon.inputs.t_min=1
    fslroi_csrecon.inputs.t_size=514

    #%%#
    #26: cat file names into list
    list_b0s_masked_csdwis = pe.Node(util.Merge(2), name='list_b0s_masked_csdwis')

    #%%#
    #27: fslmerge the b0s masked with fslroi_csrecon dwis
    merge_b0s_masked_csdwis = pe.Node(fsl.Merge(), name='merge_b0s_masked_csdwis')
    merge_b0s_masked_csdwis.inputs.dimension = 't'
    merge_b0s_masked_csdwis.inputs.output_type = 'NIFTI_GZ'


    
    #%%#
    #28: reduced data
    unique_dwis = pe.Node(interface=util.Function(
            input_names=['dsidata', 'bvals','bvecs'],
            output_names=['reduced_bvals','reduced_bvecs', 'reduced_data_file'],
            function=unique_dwis_dsi),name='unique_dwis')
    
    
    #%% collect outputs
    datasink = pe.Node(interface=DataSink(), name='datasinker')
    datasink.inputs.parameterization=True
    
    #subsitute _subj_ids_{ID} from outpaths
    subsgen = pe.Node(interface=util.Function(
            input_names=['subj_ids'],
            output_names=['substitutions'],
            function=get_subs), name='subsgen')
    
    # %% workflow connections
    dwiwf.connect(inputnode    , 'subject_ids',      subsgen,  'subj_ids')
    dwiwf.connect(subsgen      , 'substitutions',    datasink, 'substitutions') 
   
    dwiwf.connect(inputnode    ,'subject_ids',      fileselector, 'subject_id')

    #step 1 
    dwiwf.connect(fileselector , 'DSI',             cropxy_dsi, 'in_file') 
    dwiwf.connect(fileselector , 'DSIR',            cropxy_dsir,'in_file')

    #step 2
    dwiwf.connect(cropxy_dsi, 'out_file',          extractbvecsbvals, 'niftifile')
    
    #step 2.1
    dwiwf.connect(cropxy_dsi  , 'out_file',           scanner2_FOV,'file_data')
    dwiwf.connect(extractbvecsbvals, 'bvecs',         scanner2_FOV,'file_bvec')
    
    #step 3
    dwiwf.connect(cropxy_dsi,    'out_file',          extractb0_AP  , 'in_dwi')
    dwiwf.connect(extractbvecsbvals, 'bvals',       extractb0_AP  , 'in_bvals')
    
    #step 4 
    dwiwf.connect(extractb0_AP            , 'b0_AP',       list_b0AP_PA, 'in1')
    dwiwf.connect(cropxy_dsir             , 'out_file',    list_b0AP_PA, 'in2')

    #step 4.1
    dwiwf.connect(list_b0AP_PA            , 'out',fslmerge_b0AP_PA, 'in_files')
    
    #step 5 
    dwiwf.connect(extractbvecsbvals      , 'bvals',create_acqparams,'in_bvals')
    
    #step 6
    dwiwf.connect(extractbvecsbvals      , 'bvals',   create_index, 'in_bvals')
    
    #step 7
    dwiwf.connect(fslmerge_b0AP_PA   , 'merged_file',         topup, 'in_file')
    dwiwf.connect(create_acqparams   , 'acqparams_file',topup, 'encoding_file')
    
    #step 8
    dwiwf.connect(topup                , 'out_corrected',    tc_bet, 'in_file')
 
    #step 9
    dwiwf.connect(tc_bet   , 'mask_file',        dilate_tc_bet_mask, 'in_file')
   

    #step 10
    dwiwf.connect(cropxy_dsi   , 'out_file',              eddy_cuda, 'in_file')
    dwiwf.connect(dilate_tc_bet_mask,  'out_file',        eddy_cuda, 'in_mask')
    dwiwf.connect(create_acqparams  , 'acqparams_file',   eddy_cuda, 'in_acqp')
    dwiwf.connect(scanner2_FOV     , 'bvecs_file',       eddy_cuda,  'in_bvec')
    dwiwf.connect(create_index    , 'index_file',       eddy_cuda,  'in_index')
    dwiwf.connect(topup           ,'out_movpar', eddy_cuda , 'in_topup_movpar')
    dwiwf.connect(topup     ,'out_fieldcoef', eddy_cuda , 'in_topup_fieldcoef')
    

    #step 11
    dwiwf.connect(eddy_cuda     , 'out_corrected',    reorder_ec_dwi, 'in_dwi')
    dwiwf.connect(extractbvecsbvals, 'bvals',        reorder_ec_dwi, 'in_bval')
    
    #step 12
    dwiwf.connect(reorder_ec_dwi     , 'b0_out_file',      b0s_mean, 'in_file')
    
    #step 13
    dwiwf.connect(b0s_mean             , 'out_file',         ec_bet, 'in_file')
    
    #step 14
    dwiwf.connect(ec_bet   , 'mask_file',        dilate_ec_bet_mask, 'in_file')
    
    #step 15
    dwiwf.connect(dilate_ec_bet_mask, 'out_file',   mask_b0s_mean, 'mask_file')
    dwiwf.connect(b0s_mean      , 'out_file',         mask_b0s_mean, 'in_file')
    
    #step 16 
    dwiwf.connect(mask_b0s_mean, 'out_file',  fast_b0s_mean_masked, 'in_files')
    
    #step 17
    dwiwf.connect(reorder_ec_dwi    , 'b0_out_file',      list_b0s_dwis, 'in1')
    dwiwf.connect(reorder_ec_dwi    , 'dwi_out_file',     list_b0s_dwis, 'in2')

    #step 18
    dwiwf.connect(list_b0s_dwis, 'out',           fslmerge_b0sdwis, 'in_files')

    #step 19
    dwiwf.connect(extractbvecsbvals, 'bvals',        sort_bvals_bvecs, 'bvals')
    dwiwf.connect(scanner2_FOV , 'bvecs_file',       sort_bvals_bvecs, 'bvecs')   

    #step 20
    dwiwf.connect(extractbvecsbvals, 'bvals', sort_eddy_rotated_bvecs, 'bvals')
    dwiwf.connect(eddy_cuda,'out_rotated_bvecs',sort_eddy_rotated_bvecs,'bvecs')
    
    #step 21
    dwiwf.connect(fslmerge_b0sdwis, 'merged_file',    dwi_gunzipper, 'in_file')
    dwiwf.connect(dilate_ec_bet_mask, 'out_file',    mask_gunzipper, 'in_file')
    dwiwf.connect(fast_b0s_mean_masked,('partial_volume_files', get_pve0),
                  csfpve_gunzipper, 'in_file')
    
    #step 22 
    dwiwf.connect(dwi_gunzipper, 'out_file',           csrecon,      'ec_file')
    dwiwf.connect(mask_gunzipper,'out_file',        csrecon,      'brain_mask')
    dwiwf.connect(csfpve_gunzipper,'out_file',         csrecon,      'csf_pve')
    dwiwf.connect(sort_bvals_bvecs,'bvals_sortb0',       csrecon,      'bvals')
    dwiwf.connect(sort_eddy_rotated_bvecs,'bvecs_sortb0',csrecon,      'bvecs')
    
    #step 22.1 
    dwiwf.connect(csrecon,
                  'out_recon514b0s_norm_bvec',swapbvecs_recon514b0s_norm,'bvecs')
    
    #step 22.2
    dwiwf.connect(csrecon,
                  'out_recon515_bvec',            swapbvecs_recon515, 'bvecs')
    
    
    #step 22.3
    dwiwf.connect(swapbvecs_recon514b0s_norm,
                  'swaped_bvecs',   scanner2_FOV_recon514b0s_norm, 'file_bvec')
    dwiwf.connect(fileselector,
                  'DSI',            scanner2_FOV_recon514b0s_norm, 'file_data')
            
    #step 22.4
    dwiwf.connect(swapbvecs_recon515,
                  'swaped_bvecs',    scanner2_FOV_recon515, 'file_bvec')
    dwiwf.connect(fileselector,
                  'DSI',             scanner2_FOV_recon515, 'file_data')          
                        
    #step 23
    dwiwf.connect(csrecon,'out_file_csrecon',    csrecon_gzipper,'in_file')
    dwiwf.connect(csrecon,'out_file_csrecon_csf',csrecon_csf_gzipper, 'in_file')

    #step 24
    dwiwf.connect(reorder_ec_dwi,'b0_out_file',      ec_b0s_masked,  'in_file')
    dwiwf.connect(dilate_ec_bet_mask,'out_file',   ec_b0s_masked,  'mask_file')

    #step 25
    dwiwf.connect(csrecon_csf_gzipper, 'out_file',fslroi_csrecon,    'in_file')

    #step 26
    dwiwf.connect(ec_b0s_masked,'out_file',     list_b0s_masked_csdwis, 'in1')
    dwiwf.connect(fslroi_csrecon,'roi_file',    list_b0s_masked_csdwis, 'in2')

    #step 27
    dwiwf.connect(list_b0s_masked_csdwis,'out',merge_b0s_masked_csdwis, 'in_files')
    
    
    #step 28
    dwiwf.connect(csrecon,       'out_recon514b0s_bval',  unique_dwis, 'bvals')
    dwiwf.connect(scanner2_FOV_recon514b0s_norm,'bvecs_file',unique_dwis, 'bvecs')
    dwiwf.connect(merge_b0s_masked_csdwis,'merged_file',unique_dwis, 'dsidata')
    
    
    #outputs
    dwiwf.connect(inputnode       , 'subject_ids',       datasink, 'container')
    dwiwf.connect(inputnode  , 'outputdir',         datasink, 'base_directory')

    dwiwf.connect(topup, 'out_movpar',        datasink , 'Topup.@topup_movpar')
    dwiwf.connect(topup, 'out_fieldcoef',  datasink , 'Topup.@topup_fieldcoef')
    dwiwf.connect(topup, 'out_corrected',  datasink , 'Topup.@topup_corrected')

    dwiwf.connect(eddy_cuda,
                  'out_movement_rms',  datasink, 'Eddy.@out_movement_rms')
    dwiwf.connect(eddy_cuda,
                  'out_outlier_report',datasink, 'Eddy.@out_outlier_report')
    dwiwf.connect(eddy_cuda,
                  'out_outlier_map',datasink, 'Eddy.@out_outlier_map')
    dwiwf.connect(eddy_cuda,
                  'out_shell_alignment_parameters',
                  datasink, 'Eddy.@out_shell_alignment_parameters')
    
    dwiwf.connect(eddy_cuda,
                  'out_shell_PE_translation_parameters',
                  datasink, 'Eddy.@out_shell_PE_translation_parameters')
    
    dwiwf.connect(eddy_cuda,
                  'out_parameter',     datasink, 'Eddy.@out_parameter')
    dwiwf.connect(eddy_cuda,
                  'out_restricted_movement_rms',
                  datasink, 'Eddy.@out_restricted_movement_rms')
    
    dwiwf.connect(eddy_cuda,
                  'out_rotated_bvecs',
                  datasink, 'Eddy.@out_rotated_bvecs')
    
    dwiwf.connect(fslmerge_b0sdwis,
                  'merged_file',       datasink, 'Eddy.@ec_reordered')
    dwiwf.connect(b0s_mean, 'out_file',          datasink, 'Eddy.@b0smean')
    dwiwf.connect(dilate_ec_bet_mask,
                  'out_file',          datasink, 'Eddy.@ec_reordered_dilmask')
    dwiwf.connect(sort_bvals_bvecs,
                  'bvals_sortb0',      datasink, 'Eddy.@bvals_sortb0')
    dwiwf.connect(sort_bvals_bvecs,
                  'bvecs_sortb0',      datasink, 'Eddy.@bvecs_sortb0')
    dwiwf.connect(sort_eddy_rotated_bvecs,
                  'bvecs_sortb0',      datasink, 'Eddy.@rotated_bvecs_sortb0')
    
    dwiwf.connect(fast_b0s_mean_masked,
                  'tissue_class_files',  datasink, 'Fast.@tissue_class_files')
    dwiwf.connect(fast_b0s_mean_masked,
                  'probability_maps',    datasink, 'Fast.@probability_maps')
    dwiwf.connect(fast_b0s_mean_masked,
                  'restored_image',      datasink, 'Fast.@restored_image')
    dwiwf.connect(fast_b0s_mean_masked,
                  'tissue_class_map',    datasink, 'Fast.@tissue_class_map')
    dwiwf.connect(fast_b0s_mean_masked,
                  'partial_volume_files',datasink,'Fast.@partial_volume_files')
    

    dwiwf.connect(scanner2_FOV_recon514b0s_norm,
                  'bvecs_file',           datasink, 'CSRecon.@bvecs514b0snorm')
    dwiwf.connect(scanner2_FOV_recon515,
                  'bvecs_file',                 datasink, 'CSRecon.@bvecs515')
    
    dwiwf.connect(unique_dwis,
                  'reduced_data_file',  datasink, 'CSRecon.@reduced_data_file')
    dwiwf.connect(unique_dwis
                  ,'reduced_bvals',     datasink, 'CSRecon.@reduced_bvals')
    dwiwf.connect(unique_dwis,
                  'reduced_bvecs',      datasink, 'CSRecon.@reduced_bvecs')
    dwiwf.connect(csrecon_csf_gzipper,
                  'out_file',           datasink, 'CSRecon.@csrecon_csf')
   


    return dwiwf
    

