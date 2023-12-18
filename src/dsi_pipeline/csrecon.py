# -*- coding: utf-8 -*-
"""
This is matlab code wrapper interface for nipype 
"""

import os
from glob import glob
from nipype.interfaces.base import File, Directory, traits, TraitedSpec, InputMultiPath
from nipype.interfaces.fsl.base import CommandLineInputSpec, CommandLine



class CSRECONInputSpec(CommandLineInputSpec):
    ec_file = File(exists= True,  desc = 'Eddy corrected b0s merged nii time series.',argstr='%s', position=0,mandatory=True)
    brain_mask = File(exists=True, desc='Brain mask', argstr='%s',position=1,mandatory=True)
    bvals = File(exists=True, desc='bvals file', argstr='%s',position=2,mandatory=True)
    bvecs = File(exists=True, desc='bvecs file', argstr='%s',position=3,mandatory=True)
    csf_pve = File(exists=True, desc='CSF PVE estimated file', argstr='%s',position=4,mandatory=True)
    seq_params = File(exists=True, desc='SeqParams file', argstr='%s',position=5,mandatory=True)
    dirs_raw = File(exists=True, desc='DIRS GRAD RAW file', argstr='%s',position=6,mandatory=True)
    dirs_jones = File(exists=True, desc='DIRS JONES file',argstr='%s', position=7,mandatory=True)
    poolsize = traits.Int('Matlab parallel pool size', argstr='%d', position=8,mandatory=True)
    csrecon_path = File(exists=True, desc='FileName of CSRecon exe script (run_mainCS.sh)')
    mcr_path = Directory(exists=True, desc='MCR path (/path/to/compiler_Runtime/v93/')

class CSRECONOutputSpec(TraitedSpec):
    #out_files_bvecs = InputMultiPath(File(exists=True), desc='Output bvecs files' )
    #out_files_bvals = InputMultiPath(File(exists=True), desc='Output bvals files' )
    out_recon514b0s_bval=File(exists=True, desc='Output 514b0s bval file')
    out_recon514b0s_bvec=File(exists=True, desc='Output 514b0s bvec file')
    out_recon514b0s_norm_bvec=File(exists=True, desc='Output 514b0s norm bvec file')
 
    out_recon515_bval=File(exists=True, desc='Output 515 bval file')
    out_recon515_bvec=File(exists=True, desc='Output 515 bvec file')
    out_recon515_norm_bvec=File(exists=True, desc='Output 515 norm bvec file')
    
    out_file_recon_idx = File(exists=True, desc='Output recon_idx txt file')
    out_file_csrecon = File(exists=True, desc='Output csrecon.nii file')
    out_file_csrecon_csf = File(exists=True, desc='Output csrecon_csf.nii file')
    

class CSRECON(CommandLine):
    input_spec = CSRECONInputSpec
    output_spec = CSRECONOutputSpec
    _cmd = 'run_main_CS.sh'
    
    def __init__(self, **inputs):
        return super(CSRECON, self).__init__(**inputs)
    
    
    def _run_interface(self, runtime):
        
        self._cmd = self.inputs.csrecon_path + ' ' + self.inputs.mcr_path + ' ' + os.getcwd()
        runtime = super(CSRECON, self)._run_interface(runtime)
        if runtime.stderr:
           self.raise_exception(runtime)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        #outputs['out_files_bvecs'] = [os.path.abspath(f) for f in glob('*.bvec')] 
        #outputs['out_files_bvals'] = [os.path.abspath(f) for f in glob('*.bval')]
        outputs['out_recon514b0s_bval']   = [os.path.abspath(f) for f in glob('*_Recon514b0s.bval')][0]
        outputs['out_recon514b0s_bvec']   = [os.path.abspath(f) for f in glob('*_Recon514b0s.bvec')][0]
        outputs['out_recon514b0s_norm_bvec']   = [os.path.abspath(f) for f in glob('*_Recon514b0s_norm.bvec')][0]

        outputs['out_recon515_bval']   = [os.path.abspath(f) for f in glob('*_Recon515.bval')][0]
        outputs['out_recon515_bvec']   = [os.path.abspath(f) for f in glob('*_Recon515.bvec')][0]
        outputs['out_recon515_norm_bvec']   = [os.path.abspath(f) for f in glob('*_Recon515_norm.bvec')][0]

        outputs['out_file_recon_idx']   = [os.path.abspath(f) for f in glob('*_recon_idx.txt')][0]
        outputs['out_file_csrecon'] = [os.path.abspath(f) for f in glob('*_CSRecon.nii')][0]
        outputs['out_file_csrecon_csf'] = [os.path.abspath(f) for f in glob('*_CSRecon_csf.nii')][0]
        return outputs
    
