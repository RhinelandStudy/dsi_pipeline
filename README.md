# dsi_pipeline
DSI processing and modeling pipeline: pipeline for DSI diffusion scans pre-processing and microstructure modeling.


## Build docker image

```bash

nvidia-docker build -t dsi_pipeline -f docker/Dockerfile .


```

## Or pull from docker hub

```bash
docker pull dznerheinlandstudie/rheinlandstudie:dsi_pipeline
```

## Run pipeline:

### Using docker
The pipeline can be run with docker by running the container as follows:


```bash

 nvidia-docker run --rm \
                 -v /path/to/input_scans:/input \
                 -v /path/to/work_folder:/work \
                 -v /path/to/output:/output \
        dznerheinlandstudie/rheinlandstudie:dsi_pipeline \
        run_dsi_pipeline \
        -s /input \
        --subjects test_subject_01 \
        -w /work \
        -o /output \ 
        -p 4 -t 2 -g 1 -gp 1

```

The command line options are described briefly if the pipeline is started with only ```-h``` option.

### Using Singulraity

The pipeline can be run with Singularity by running the singularity image as follows:

```bash


singularity build dsi_pipeline.sif docker://dznerheinlandstudie/rheinlandstudie:dsi_pipeline
```

When the singularit image is created, then it can be run as follows:

```bash

PREPROCESSING:

singularity run --nv -B /path/to/inputdata:/input \
                     -B /path/to/work:/work \
                     -B /path/to/output:/output \
            dsi_pipeline.sif "export OPENBLAS_NUM_THREADS=1;export GOTO_NUM_THREADS=1;export OMP_NUM_THREADS=1;ulimit -s unlimited;export PYTHONWARNINGS="ignore";\
            run_dsi_pipeline \ 
                      -s /input \
                      -w /work \
                      -o /output \ 
                      -p 2 -g 1 -gp 1 \
                      --csrecon /opt/CSRecon_deploy/run_main_CS.sh \
                      --mcr /opt/MCR2017b/v93 \
                      --poolsize 2 \
                      --subjects test_subject_01"

MODELING:

singularity run --nv -B /path/to/work:/work_mod \
                     -B /path/to/output:/output \
                     "export OPENBLAS_NUM_THREADS=1;export GOTO_NUM_THREADS=1;export OMP_NUM_THREADS=1;\
                     ulimit -s unlimited;export PYTHONWARNINGS="ignore";\
                     run_dsi_modeling  \
                        -w /work_mod \
                        -o /output \
                        -i /output \
                        -p 2 -g 1 -gp 1 -mp 1 \
                        --subjects test_subject_01"

```



