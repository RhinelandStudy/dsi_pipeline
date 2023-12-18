# -*- coding: utf-8 -*-
"""
This is matlab code wrapper interface for nipype 
Copyright 2023 Population Health Sciences, German Center for Neurodegenerative Diseases (DZNE)
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0 
Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

"""
from __future__ import print_function
import os
from glob import glob
from nipype.interfaces.base import File, Directory, traits, TraitedSpec, InputMultiPath
from nipype.interfaces.fsl.base import CommandLineInputSpec, CommandLine



class MDTGenProtocolInputSpec(CommandLineInputSpec):
    
    bvecs = File(exists=True, desc='bvecs file', argstr='%s',position=0,mandatory=True)
    bvals = File(exists=True, desc='bvals file', argstr='%s',position=1,mandatory=True)
    prtcl_name = traits.String('output protocol name', argstr='-o %s', position=2,mandatory=True)
    seq_params_file = File(exists=True, desc='SeqParams file', mandatory=True)
    
class MDTGenProtocolOutputSpec(TraitedSpec):
    #out_files_bvecs = InputMultiPath(File(exists=True), desc='Output bvecs files' )
    #out_files_bvals = InputMultiPath(File(exists=True), desc='Output bvals files' )
    out_prtcl=File(exists=True, desc='Output generated protocol file')
    

class MDTGenProtocol(CommandLine):
    input_spec = MDTGenProtocolInputSpec
    output_spec = MDTGenProtocolOutputSpec
    _cmd = 'mdt-generate-protocol'
    
    def __init__(self, **inputs):
        return super(MDTGenProtocol, self).__init__(**inputs)
    
    
    def _run_interface(self, runtime):
        params=None
        with open(self.inputs.seq_params_file, 'r') as fp:
            params=fp.readlines()
        bigDel  = params[0].split('\t')[1].strip()
        smallDel= params[0].split('\t')[2].strip()
        TE      = params[0].split('\t')[3].strip()
        
        args = " --sequence-timing-units 's' --Delta %s --delta %s --TE %s" % (bigDel, smallDel,TE)

        self._cmd = self._cmd  + args 

        runtime = super(MDTGenProtocol, self)._run_interface(runtime)
        if runtime.stderr:
           self.raise_exception(runtime)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        #outputs['out_files_bvecs'] = [os.path.abspath(f) for f in glob('*.bvec')] 
        #outputs['out_files_bvals'] = [os.path.abspath(f) for f in glob('*.bval')]
        outputs['out_prtcl']   = os.path.abspath(self.inputs.prtcl_name)

        return outputs
    

class MDTFitModelInputSpec(CommandLineInputSpec):
    MDT = traits.String(desc='Model name', argstr='"%s"', position=0,mandatory=True)
    data_file = File(exists= True,  desc = 'CSrecon dwi nii time series.',argstr='%s', position=1,mandatory=True)
    prtcl_file = File(exists=True, desc='mdt protocol file', argstr='%s', position=2, mandatory=True)
    brain_mask = File(exists=True, desc='Brain mask', argstr='%s',position=3,mandatory=True)
    out_dir = traits.String(desc='Output directory name', argstr='-o %s', position=4,mandatory=True)
    dev_ind = traits.String(desc='CL device index', argstr='--cl-device-ind %s', position=5)
    noise_std = traits.File(desc='the noise std, defaults to None for automatic noise estimation',argstr='--noise-std %s',position=6,mandatory=False)
    use_gpu = traits.Bool(desc='set  the flag to use gpu')
    
class MDTFitModelOutputSpec(TraitedSpec):
    outputfolder = InputMultiPath(Directory(exists=True), desc='Output files folder' ) 

class MDTFitModel(CommandLine):
    input_spec = MDTFitModelInputSpec
    output_spec = MDTFitModelOutputSpec
    _cmd = 'mdt-model-fit'
    
    def __init__(self, **inputs):
        return super(MDTFitModel, self).__init__(**inputs)
    
    def _format_arg(self, name, spec, value):
        if name=='dev_ind':
            print("dev_ind value passed is: %s" % value)
            try:
                gpuID=os.environ['CUDA_VISIBLE_DEVICE']
                print("But, found cuda visible device from the env, will use it...")
            except:
                gpuID="0"
                print("No cuda visible device index found, will use id 0.")
            return spec.argstr %( gpuID )
        if name=='out_dir':
            self.outputfolder_name = self.inputs.out_dir + (self.inputs.MDT.replace(' ','').replace('(','').replace(')',''))
            print("setting output directory name to %s" % self.outputfolder_name)
            return spec.argstr % (self.outputfolder_name)
        
        return super(MDTFitModel, self)._format_arg(name, spec, value)
 
    def _run_interface(self, runtime):
        runtime = super(MDTFitModel, self)._run_interface(runtime)
        #self.inputs.out_dir=os.getcwd()
        if runtime.stderr:
           self.raise_exception(runtime)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['outputfolder']          = os.path.abspath(self.outputfolder_name)

        return outputs
