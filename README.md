# Diffusion MRI preprocessing and modeling pipeline
This repository contains a Nipype wrapper for the diffusion MRI processing pipeline that is used to process the diffusion spectrum imaging (DSI) scans acquired with the MRI protocol in the Rhineland Study. The pipeline consists of the preprocessing including motion and geometric distortion correction and compressed sensing reconstruction, and the diffusion and microstructure modeling fitting the diffusion tensor, diffusion kurtosis and NODDI diffusion models to the data.

If you use this wrapper please cite:
Tobisch A, Stirnberg R, Harms RL, Schultz T, Roebroeck A, Breteler MMB and Stöcker T (2018) Compressed Sensing Diffusion Spectrum Imaging for Accelerated Diffusion Microstructure MRI in Long-Term Population Imaging. Front. Neurosci. 12:650. https://doi.org/10.3389/fnins.2018.00650

```
@article{tobisch2018,
  title={Compressed Sensing Diffusion Spectrum Imaging for Accelerated Diffusion Microstructure MRI in Long-Term Population Imaging},      
  author={Tobisch, Alexandra and Stirnberg, Rüdiger and Harms, Robbert L. and Schultz, Thomas and Roebroeck, Alard and Breteler, Monique M. B. and Stöcker, Tony},   
	journal={Frontiers in Neuroscience},      
	volume={12},           
	year={2018},
  DOI={10.3389/fnins.2018.00650}
}
```
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
            dsi_pipeline.sif "export OPENBLAS_NUM_THREADS=1;export GOTO_NUM_THREADS=1;\
            export OMP_NUM_THREADS=1;ulimit -s unlimited;export PYTHONWARNINGS='ignore';\
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
            dsi_pipeline.sif "export OPENBLAS_NUM_THREADS=1;export GOTO_NUM_THREADS=1;\
            export OMP_NUM_THREADS=1;ulimit -s unlimited;export PYTHONWARNINGS='ignore';\
                     run_dsi_modeling  \
                        -w /work_mod \
                        -o /output \
                        -i /output \
                        -p 2 -g 1 -gp 1 -mp 1 \
                        --subjects test_subject_01"

```
