# -*- coding: utf-8 -*-
"""
This is FSL tensor decomposition interface for nipype
Copyright 2023 Population Health Sciences, German Center for Neurodegenerative Diseases (DZNE)
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0 
Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

"""

import os
from nipype.interfaces.base import File, Directory, traits, TraitedSpec, InputMultiPath, isdefined
from nipype.interfaces.fsl.base import CommandLineInputSpec, CommandLine,FSLCommand, FSLCommandInputSpec

class FSLMathsInput(FSLCommandInputSpec):

    in_file = File(position=0, argstr="%s", exists=True, mandatory=True,
                   desc="image to operate on")
    out_basename = traits.String(position=-1, argstr="%s",
                    desc="image to write", hash_files=False,mandatory=True)
    operation = traits.Enum("tensor_decomp", argstr="-%s", position=1, mandatory=True,
                            desc="operation to perform")


class FSLMathsOutput(TraitedSpec):

    out_file = File(exists=True, desc="image written after calculations")
    L1 = File(exists=True, desc='path/name of file with the 1st eigenvalue')
    L2 = File(exists=True, desc='path/name of file with the 2nd eigenvalue')
    L3 = File(exists=True, desc='path/name of file with the 3rd eigenvalue')
    FA = File(exists=True, desc='path/name of file with the fractional anisotropy')
    MD = File(exists=True, desc='path/name of file with the mean diffusivity')
    MO = File(exists=True, desc='path/name of file with the mode of anisotropy')
    V1 = File(exists=True, desc='path/name of file with the 1st eigenvector')
    V2 = File(exists=True, desc='path/name of file with the 2nd eigenvector')
    V3 = File(exists=True, desc='path/name of file with the 3rd eigenvector')

class FSLMathsCommand(FSLCommand):

    _cmd = "fslmaths"
    input_spec = FSLMathsInput
    output_spec = FSLMathsOutput

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for output in outputs:
            if output=='out_file':
                outputs["out_file"] =  os.path.abspath(self.inputs.out_basename + '.nii.gz')
            else:
                outputs[output] = os.path.abspath(self.inputs.out_basename + '_' + output + '.nii.gz')
        return outputs

