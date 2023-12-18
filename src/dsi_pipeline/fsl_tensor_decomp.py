# -*- coding: utf-8 -*-
"""
This is FSL tensor decomposition interface for nipype
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

