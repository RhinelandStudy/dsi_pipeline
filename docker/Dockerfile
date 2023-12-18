FROM centos:7

#----------------------------
# Install common dependencies
#----------------------------
RUN yum install -y epel-release bzip2 ca-certificates curl perl unzip bc  gcc gcc-c++ libXt libXext libXrender libXtst libquadmath-devel postgresql-devel tcsh \
    && yum install -y dc mono-mcs opencl-headers clinfo openblas openblas-devel \
    && yum clean packages \
    && rm -rf /var/cache/yum/* /tmp/* /var/tmp/*

#-------------------------------------------------
# Set paths and evn vars 
#-------------------------------------------------
ENV PATH=/opt:/opt/miniconda/envs/default/bin:/opt/fsl/bin:/usr/local/cuda/bin:/usr/local/nvidia:/usr/local/nvidia/bin:$PATH \
    FSLDIR=/opt/fsl \
    FREESURFER_HOME=/opt/freesurfer \ 
    ANTSPATH=/opt/ants \
    FSLOUTPUTTYPE='NIFTI_GZ' \
    LD_LIBRARY_PATH=/opt/fsl/lib:/usr/local/nvidia:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/lib:/.singularity.d/libs:/lib64:$LD_LIBRARY_PATH \
    MCR_INHIBIT_CTF_LOCK=1 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    PATH=$FREESURFER_HOME/bin:$ANTSPATH/bin:$PATH \
    STARTUP="/opt/startup.sh"

#-------------------------------------------------
# Install Miniconda, and set up Python 3 environment
#-------------------------------------------------
RUN echo "Downloading Miniconda installer ..." \
    && curl -sSL -o miniconda.sh http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/miniconda \
    && rm -f miniconda.sh \
    && /opt/miniconda/bin/conda config --add channels conda-forge \
    && /opt/miniconda/bin/conda create -y -n default python=3.7 cython pyopencl \
    && echo "libnvidia-opencl.so.1" > /opt/miniconda/envs/default/etc/OpenCL/vendors/nvidia.icd \
    && source /opt/miniconda/bin/activate default \
    && pip install -U --no-cache-dir pip \
    && pip install --no-cache-dir nipype==1.4.2 dcmstack==0.7.0 \
    && pip install --no-cache-dir mot==0.9.4 \
    && pip install --no-cache-dir mdt==1.1.0 \
    && pip install --no-cache-dir niflow-nipype1-workflows==0.0.4 pyOpenSSL==19.0.0 GPUtil==1.3.0 dipy==1.1.1 matplotlib==2.2.3 psycopg2==2.7.3.2 pyxnat==1.2 pycrypto==2.6.1 scikit-image==0.14.5 \
    && conda clean -y --all \
    && ln -sf /opt/miniconda/envs/default/bin/mdt-create-protocol /opt/miniconda/envs/default/bin/mdt-generate-protocol 


RUN echo "Installing CUDA toolkit 8..." \
    && yum install -y wget \
    && echo "downloading cuda toolkit..." \
    && wget -q https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run \
    && chmod a+x cuda_8.0.61_375.26_linux-run \
    && ./cuda_8.0.61_375.26_linux-run --extract=/tmp \
    && /tmp/cuda-linux64-rel-8.0.61-21551265.run -noprompt \
    && rm -f cuda* /tmp/cuda* /tmp/NVIDIA* 

COPY ./src /opt/

RUN echo "Getting Freesurfer binaries..." \
    && cd /opt \
    && curl -sSL --retry 5 https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.0/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz | gunzip | tar -x -v --files-from fs_binaries.txt -f -

RUN echo "Getting FSL binaries..." \
    && yum install -y make zlib-devel \
    && wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.10-sources.tar.gz \
    && tar zxf fsl-5.0.10-sources.tar.gz \
    && cd fsl \
    && ./build || true  \
    && cd / \
    && chmod a+x /opt/copy_bin.sh && ./opt/copy_bin.sh \
    && mv -f /opt/b02b0_3.cnf /opt/fsl/etc/flirtsch/ \
    && wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/patches/eddy-patch-fsl-5.0.11/centos6/eddy_cuda8.0 \
    && mv -f eddy_cuda8.0 /opt/fsl/bin && ln -s /opt/fsl/bin/eddy_cuda8.0 /opt/fsl/bin/eddy_cuda && chmod a+x /opt/fsl/bin/* \
    && rm -rf fsl-5.0.10-sources.tar.gz && rm -rf /fsl

RUN echo "Install MCR 2017b..." \
    && curl -sSL -o MCR_R2017b_glnxa64_installer.zip https://ssd.mathworks.com/supportfiles/downloads/R2017b/deployment_files/R2017b/installers/glnxa64/MCR_R2017b_glnxa64_installer.zip \
    && unzip -d MCR_R2017b MCR_R2017b_glnxa64_installer.zip \
    && rm -f MCR_R2017b*.zip \
    && mkdir /opt/MCR2017b \
    && cd MCR_R2017b \
    && echo "destinationFolder=/opt/MCR2017b" > installer_input.txt \
    && echo "agreeToLicense=yes" >> installer_input.txt \
    && echo "mode=silent" >> installer_input.txt \
    && ./install -inputFile /MCR_R2017b/installer_input.txt \
    && cd .. && rm -rf MCR_R2017b


RUN echo "Updating settings..." \
    && chmod -R a+rx /opt/CSRecon_deploy/* \
    && chmod -R a+rx /opt/dsi_pipeline \
    && source /opt/miniconda/bin/activate default \
    && cd /opt \
    && pip install -e . \
    && cp -f /opt/dsi_pipeline/multiproc.py /opt/miniconda/envs/default/lib/python3.7/site-packages/nipype/pipeline/plugins/ \
    && cp -f /opt/dsi_pipeline/epi.py /opt/miniconda/envs/default/lib/python3.7/site-packages/nipype/interfaces/fsl/ \
    && cp -rf /opt/dsi_pipeline/fslpython/fsl /opt/miniconda/envs/default/lib/python3.7/site-packages/ \
    && chmod a+r /opt/fsl/etc/flirtsch/* \
    && mkdir -p ~/.config/matplotlib \
    && echo 'backend: Agg' > ~/.config/matplotlib/matplotlibrc \
    && sed -i 's/backend      : TkAgg/backend      : Agg/' /opt/miniconda/envs/default/lib/python3.7/site-packages/matplotlib/mpl-data/matplotlibrc \
    && if [ ! -f "$STARTUP" ]; then \
       echo '#!/usr/bin/env bash' >> $STARTUP; \
       echo 'source  /docker_environment' >> $STARTUP; \
       echo 'export LD_LIBRARY_PATH=/.singularity.d/libs:/lib64:$LD_LIBRARY_PATH' >> $STARTUP; \
       echo 'source /opt/miniconda/bin/activate default' >> $STARTUP; \
       echo 'export PATH=$FREESURFER_HOME/bin:$FSLDIR/bin:$ANTSPATH/bin:$PATH' >> $STARTUP; \
       echo 'mdt-init-user-settings' >> $STARTUP; \
       echo 'IFS=";" read -ra COMMANDS <<< "$@"' >> $STARTUP; \
       echo 'for command in "${COMMANDS[@]}";do' >> $STARTUP; \
       echo 'echo "`date` : executing ${command}";' >> $STARTUP; \
       echo '${command}' >> $STARTUP; \
       echo 'done' >> $STARTUP; \
       fi \
    && chmod 755 $STARTUP

ENTRYPOINT ["/opt/startup.sh"]
