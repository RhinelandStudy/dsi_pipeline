#!/usr/bin/env python

"""
#Rhineland Study MRI Post-processing pipelines
#rs_dsi_pipeline: Diffusion DSI scans processing pipeline using FSL and nipype
"""
import os
import sys
from glob import glob
if os.path.exists('MANIFEST'): os.remove('MANIFEST')


def main(**extra_args):
    from setuptools import setup
    setup(name='dsi_pipeline',
          version='1.0.0',
          description='RhinelandStudy Diffusion DSI Pipeline',
          long_description="""RhinelandStudy processing for diffusion DSI scans """ + \
          """It also offers support for performing additional options to run post processing analyses.""" + \
          """More pipelines addition is work in progress.""",
          author= 'shahidm',
          author_email='mohammad.shahid@dzne.de',
          url='http://www.dzne.de/',
          packages = ['dsi_pipeline'],
          entry_points={
            'console_scripts': [
                             "run_dsi_pipeline=dsi_pipeline.run_dsi_pipeline:main",
                             "run_dsi_modeling=dsi_pipeline.run_dsi_modeling:main",
                              ]
                       },
          license='DZNE License',
          classifiers = [c.strip() for c in """\
            Development Status :: 1 
            Intended Audience :: Developers
            Intended Audience :: Science/Research
            Operating System :: OS Independent
            Programming Language :: Python
            Topic :: Software Development
            """.splitlines() if len(c.split()) > 0],    
          maintainer = 'RheinlandStudy MRI/MRI-IT group, DZNE',
          maintainer_email = 'mohammad.shahid@dzne.de',
          package_data = {'dsi_pipeline':
		['*.bval','*.txt','FSL_labels/*.gz','FSL_labels/optPar/*.cnf']},
          install_requires=["nipype","dcmstack","nibabel"],
          **extra_args
         )

if __name__ == "__main__":
    main()

