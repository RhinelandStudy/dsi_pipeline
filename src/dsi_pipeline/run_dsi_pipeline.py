#!/usr/bin/env python

"""
Copyright 2023 Population Health Sciences, German Center for Neurodegenerative Diseases (DZNE)
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0 
Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""

from __future__ import print_function

from .dsi_pipeline import create_dsi_pipeline

from nipype import config, logging

import os, sys,glob
import argparse
from itertools import chain

def create_dsi_wf(scans_dir, work_dir, outputdir,subject_ids,
                  csrecon_path,mcr_path,poolsize,  wfname='fs_pipeline'):
    dsiwf = create_dsi_pipeline(scans_dir, work_dir, outputdir, subject_ids,
                                csrecon_path,mcr_path, poolsize, 
                                wfname)
    dsiwf.inputs.inputnode.subject_ids = subject_ids
    
    return dsiwf
    
    
def main():
    """
    Command line wrapper for preprocessing data
    """
    parser = argparse.ArgumentParser(description='Run DSI pipeline for DSI '
                                     'dMRI diffusion data.',epilog='Example-1:'
                                     '{prog} -s /data/scans -w /data/workdir'
                                     '-p 2 -t 2 --subjects subj1 subj2 -o '
                                     '/data/dsi_output'
                                     '\nExample-2: {prog} -s /data/scans -w /data/work'
                                     '-o /data/output -p 64 --poolsize 32 \n\n'
                                     .format(prog=os.path.basename(
                                             sys.argv[0])),
                                             formatter_class=argparse.
                                             RawTextHelpFormatter)

    parser.add_argument('-s', '--scansdir', help='Scans directory where data'
                        ' is already downloaded for each subject',
                        required=True)
    
    parser.add_argument('-w', '--workdir', help='Work directory where data'
                        ' is processed by pipeline for each subject.',
                        required=True)

    parser.add_argument('-o', '--outputdir', help='Output directory where '
                        'final results will be saved.', required=True)

    parser.add_argument('--subjects', help='One or more subject IDs'
                        '(space separated).',
                        default=None, required=False, nargs='+',
                        action='append')
    
    parser.add_argument('-b', '--debug', help='Debug mode', action='store_true')
    
    parser.add_argument('-g', '--ngpus', help='Number of GPUs to use in '
                        'parallel', default=1, type=int)
    
    parser.add_argument('-gp', '--ngpuproc', help='Number of processes per GPU',
                        default=1, type=int)
    
    parser.add_argument('-p', '--processes', help='Overall number of (nipype) '
                        'parallel processes', default=1, type=int)
        
    parser.add_argument('-t', '--threads', help='ITK threads', default=1,
                        type=int)
    
    parser.add_argument('-n', '--name', help='Pipeline workflow name',
                        default='dsi_pipeline')
    
    parser.add_argument('--csrecon',help='CSRecon executable shell script '
                        '(/opt/CSRecon_deploy/run_main_CS.sh)',
                        required=False,
                        default='/opt/CSRecon_deploy/run_main_CS.sh')
    
    parser.add_argument('--mcr',
                        help='Matlab Compiler Runtime (/opt/MCR2017b/v93)',
                        required=False, default='/opt/MCR2017b/v93/')
    
    parser.add_argument('--poolsize', help='Matlab pool size', required=False,
                        type=int, default=12)
    
    
    args = parser.parse_args()
    
    scans_dir = os.path.abspath(os.path.expandvars(args.scansdir))
    
    if not os.path.exists(scans_dir):
        raise IOError("Scans directory does not exist.")
        
    
    subject_ids = []
    
    if args.subjects:
        subject_ids = list(chain.from_iterable(args.subjects))
    else:
        subject_ids = glob.glob(scans_dir.rstrip('/') +'/*')
        subject_ids = [os.path.basename(s.rstrip('/')) for s in subject_ids]


    print("Creating DSI Pipeline workflow...")
    work_dir = os.path.abspath(os.path.expandvars(args.workdir))
    outputdir = os.path.abspath(os.path.expandvars(args.outputdir))
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)        
    
    config.update_config({
        'logging': {'log_directory': args.workdir, 'log_to_file': True},
        'execution': {'job_finished_timeout' : 65,
                      'poll_sleep_duration' : 30,
                      'hash_method' : 'content',
                      'local_hash_check' : True,
                      'stop_on_first_crash':False,
                      'crashdump_dir': args.workdir,
                      'crashfile_format': 'txt'
                       },
                       })

    #config.enable_debug_mode()
    logging.update_logging(config)
    

    dsi_pipeline = create_dsi_wf(scans_dir, work_dir, outputdir, subject_ids,
                                 args.csrecon, args.mcr,
                                 args.poolsize, wfname='dsi_pipeline')
        
    # Visualize workflow
    if args.debug:
        dsi_pipeline.write_graph(graph2use='flat', simple_form=True)

        
    dsi_pipeline.run(
            plugin='MultiProc', 
            plugin_args={'n_procs' : args.processes, 'n_gpus': args.ngpus,
                         'ngpuproc': args.ngpuproc}
            )
    

    print('Done DSI pipeline. Please check results in %s directory' %outputdir)
    

    
if __name__ == '__main__':
    sys.exit(main())
