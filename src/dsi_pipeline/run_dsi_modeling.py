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

from .dsi_modeling import create_dsi_modeling

from nipype import config, logging

import os, sys,glob
import argparse
from itertools import chain

def create_dsi_mod_wf(input_dir, work_dir, outputdir,subject_ids, mdtproc, wfname='dsi_modeling'):
    dsi_wf = create_dsi_modeling(input_dir, work_dir, outputdir, subject_ids, mdtproc, wfname)
    dsi_wf.inputs.inputnode.subject_ids = subject_ids
    return dsi_wf


def main():
    """
    Command line wrapper for preprocessing data
    """
    parser = argparse.ArgumentParser(description='Run DSI postproc for dMRI '\
                                     'preprocessed diffusion data.',
                                     epilog='Example-1: {prog} -i '\
                                     '/data/dsi_output -w '\
                                     '/data/dsi_work -o /data/dsi_output -p 2 -t 2 '\
                                     '--subjects subj1 subj2 ... '\
                                     '\nExample-2: {prog} -i /data/dsi_output -w /data/dsi_work -o'\
                                     ' /data/dsi_output -p 2 -t 2 -g 1 -gp 2 -mp 1'\
                                     '\n\n'
                                     .format(prog=os.path.basename\
                                             (sys.argv[0])),\
                                     formatter_class=argparse.\
                                     RawTextHelpFormatter)

    parser.add_argument('-i', '--inputdir', help='Input directory where data' \
                        ' is already preprocessed for each subject.', required=True)
    parser.add_argument('-w', '--workdir', help='Work directory where data' \
                        ' is processed for each subject.', required=True)

    parser.add_argument('-o', '--outputdir', help='Output directory where ' \
                        'results will be stored.', required=True)

    parser.add_argument('--subjects', help='One or more subject IDs'\
                        '(space separated).', \
                        default=None, required=False, nargs='+', action='append')
    parser.add_argument('-b', '--debug', help='debug mode', action='store_true')
    parser.add_argument('-p', '--processes', help='overall number of (nipype) parallel processes', \
                        default=1, type=int)
    parser.add_argument('-t', '--threads', help='ITK threads', default=1,\
                        type=int)

    parser.add_argument('-g', '--ngpus', help='Number of GPUs to use in '
                        'parallel', default=1, type=int)

    parser.add_argument('-gp', '--ngpuproc', help='Number of processes per GPU',
                        default=1, type=int)

    parser.add_argument('-mp', '--mdtproc', help='MDT nodes n_proc. This will '
                        'be used to control MDT jobs on GPU slots.',
                        default=1, type=int)

    parser.add_argument('-n', '--name', help='Pipeline workflow name',
                        default='dsi_modeling')

    args = parser.parse_args()

    input_dir = os.path.abspath(os.path.expandvars(args.inputdir))
    if not os.path.exists(input_dir):
        raise IOError("Input directory does not exist.")


    subject_ids = []

    if args.subjects:
        subject_ids = list(chain.from_iterable(args.subjects))
    else:
        subject_ids = glob.glob(input_dir.rstrip('/') +'/*')
        subject_ids = [os.path.basename(s.rstrip('/')) for s in subject_ids]


    print("Creating dsi modeling workflow...")
    work_dir = os.path.abspath(os.path.expandvars(args.workdir))
    outputdir = os.path.abspath(os.path.expandvars(args.outputdir))

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    config.update_config({
        'logging': {'log_directory': args.workdir, 'log_to_file': True},
        'execution': {'job_finished_timeout' : 25,
                      'poll_sleep_duration' : 10,
                      'hash_method' : 'content',
                      'local_hash_check' : True,
                      'stop_on_first_crash':False,
                      'crashdump_dir': args.workdir,
                      'crashfile_format': 'txt'
                       },
                       })

    #config.enable_debug_mode()
    logging.update_logging(config)


    dsi_modeling = create_dsi_mod_wf(input_dir, work_dir, outputdir, subject_ids, args.mdtproc, wfname='dsi_modeling')

    # Visualize workflow
    if args.debug:
        dsi_modeling.write_graph(graph2use='flat', simple_form=True)


    #print('\nPipelines workflow will run the following pipelines: %s' %
    #      (args.run_pipeline))

    dsi_modeling.run(
            plugin='MultiProc',
            plugin_args= {'n_procs' : args.processes, 'n_gpus': args.ngpus,
                          'ngpuproc': args.ngpuproc}
            )


    print('Done DSI modeling!!!')


if __name__ == '__main__':
    sys.exit(main())
