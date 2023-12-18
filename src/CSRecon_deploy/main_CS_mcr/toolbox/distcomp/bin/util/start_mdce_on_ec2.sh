#!/bin/sh
#
# Copyright 2010-2016 The MathWorks, Inc.
#
# This script runs at EC2 instance startup, interprets its environment, and
# turns the instance into a HEADNODE or WORKER. 
#
# This script checks that environment variables it depends on are supplied.
# It DOES NOT CHECK that environment variables it does not use are supplied.
# It DOES NOT CHECK that environment variables required for licensing are supplied.
#
# The current environment variables used are:
#
# ----------------
# HEADNODE_NAME
# ----------------
# * Required
# * Default:  HEADNODE_NAME=
# This specifies where to find the headnode. Use the EC2 public name.
#
# Example:
# HEADNODE_NAME=ec2-75-101-238-144.compute-1.amazonaws.com
#
# ----------------
# MDCEOPTS
# ----------------
# * Optional
# * Default:  MDCEOPTS=
# These are additional options or flags to start MDCE via commandline with.
# 
# Example:
# MDCEOPTS=-securitylevel 2
#
# ----------------
# NUM_WORKERS
# ----------------
# * Optional
# * Default:  NUM_WORKERS=8
# Specifies how many workers to be started on the current machine.  This defaults to 8, because
# whole EC2 machines have 8 cores.  For memory intensive problems, it might be beneficial
# to decrease the number of workers per machine. For IO intensive problems, it
# might be beneficial to increase the number of workers per machine.
# 
# Example:
# NUM_WORKERS=4
#
#
# ----------------
# MDCS_BIN
# ----------------
# * Required
#
# Path to (matlabroot)/toolbox/distcomp/bin.
#
# Example:
# export MDCS_BIN=/mnt/matlab/toolbox/distcomp/bin
#
# ----------------
# JOB_MANAGER_NAME
# ----------------
# * Optional
# * Default: EC2_job_manager
# Specifies the name to use for the Mathworks Job Manager.
#
# Example:
# export JOB_MANAGER_NAME=your_job_manager
#
# ----------------
# MDCE_EPHEMERAL_BASE
# ----------------
# * Optional
# * Default: /mnt/mdce
# Specifies the directory to use for mdce's ephemeral work
#
# Example:
# export MDCE_EPHEMERAL_BASE=/path/youd/like/for/mdce/ephemeral
#
# ----------------
# MDCE_LOG_PATH
# ----------------
# * Optional
# * Default: ${MDCE_EPHEMERAL_BASE}/log
# Specifies the directory to use for mdce's logs
#
# Example:
# export MDCE_LOG_PATH=/path/youd/like/for/mdce/logs
#
# ----------------
# MDCE_TMP
# ----------------
# * Optional
# * Default: /shared/tmp/mdceworkertmp
# Specifies the directory to use for mdce worker temporary files -- Matlab's tempdir.
#
# Example:
# export MDCE_TMP=/path/youd/like/for/worker/tmp
#
# ----------------
# MDCS_MIN_PEER_MESSAGING_PORT
# ----------------
# * Optional
# * Default: 14350
#
# ----------------
# 
# ----------------
# MDCS_MAX_PEER_MESSAGING_PORT
# ----------------
# * Optional
# * Default: MDCS_MIN_PEER_MESSAGING_PORT+1 + 4*NUM_WORKERS
#
# ---------------
# MDCS_SCRIPT_ROOT
# ---------------
# * Required
#
# ----------------
# ENABLE_MATLAB_DRIVE
# ----------------
# * Optional
# * Default: ENABLE_MATLAB_DRIVE=0
# Set ENABLE_MATLAB_DRIVE=1 to enable MATLAB Drive integration on mdce workers.
# Note: In addition to setting ENABLE_MATLAB_DRIVE=1, you will need to use
# ${MDCS_BIN}/util/mount_gds.sh to mount the MATLAB Drive folder.
#
##########################
#
# Exit codes
#
# 0  Success
# 1  Missing environment variable
# 2  Could not start mdce
# 3  Could not start jobmanager
# 4  Could not start a worker
#
###################################################################

########################################
#
#   Functions
#
########################################

# wrapper for echo that takes into account the verbose flag
verbose() {
    if [ -z "${VERBOSE}" ] || [ "${VERBOSE}" -eq 1 ]; then
        echo $1
    fi
}

cleanMdcsStart() {

    if [ -z "${MLM_LICENSE_FILE}" ]; then
    WEB_LICENSE_FLAGS=" -requireweblicensing -ondemand"
    else
    WEB_LICENSE_FLAGS=""
    fi

    if [ "${ENABLE_MATLAB_DRIVE}" -eq 1 ]; then
    MATLAB_DRIVE_FLAGS=" -enableMATLABDrive \
        -matlabDriveFolderLocationCfg ${MDCE_TMP}/MATLAB-cloud-storage-location"
    else
    MATLAB_DRIVE_FLAGS=""
    fi

    MDCE_COMMAND="${MDCS_BIN}/mdce \
        -cleanPreserveJobs \
        -logbase ${MDCE_LOG_PATH} \
        -checkpointbase ${MDCE_EPHEMERAL_BASE}/checkpoint \
        -pidbase ${MDCE_EPHEMERAL_BASE}/pid \
        -allserversocketsincluster \
        -workerproxiespoolconnections \
        -enablepeerlookup \
        -sendactivitynotifications \
        -usesecurecommunication \
        -untrustedclients \
        -sharedsecretfile ${KEYSTORE_FILE} \
        -hostname ${MDCS_PUBLIC_HOSTNAME} \
        -loglevel ${MDCE_LOG_LEVEL} \
        ${WEB_LICENSE_FLAGS} \
        ${MATLAB_DRIVE_FLAGS} \
        ${MDCEOPTS}"

    # if unable to write the stop script, the consequences are very mild. Just keep going.
    mkdir -p ${MDCE_EPHEMERAL_BASE}/bin

    STOP_MDCE_SCRIPT=${MDCE_EPHEMERAL_BASE}/bin/stop_mdce_on_ec2.sh
    WAIT_COMMAND=${MDCS_BIN}/util/waitForProcessesWithNameToEnd.sh

    echo "#!/bin/sh"                           > ${STOP_MDCE_SCRIPT}
    echo "${MDCE_COMMAND} stop"               >> ${STOP_MDCE_SCRIPT}
    echo "${WAIT_COMMAND} session_helper 300" >> ${STOP_MDCE_SCRIPT}

    chmod 777 ${STOP_MDCE_SCRIPT}

    # Ensure clean restart of MDCS services
    ${MDCE_COMMAND} stop
    rm -rf ${MDCE_LOG_PATH}
    rm -rf ${MDCE_EPHEMERAL_BASE}/pid

    ${MDCE_COMMAND} start
   
    if [ $? -ne 0 ]; then
        echo "MDCE startup failed. Check ${MDCE_LOG_PATH} for details."
        exit 2
    fi
}


startWorkers() {
    MIN_NUM_WORKERS=8

    if [ -z "${NUM_WORKERS}" ]; then
        NUM_WORKERS=${MIN_NUM_WORKERS}
    fi

    JMHOST=${HEADNODE_NAME}
    verbose "Job manager host: ${JMHOST}"

    WORKER_NAME=${MDCS_PUBLIC_HOSTNAME}_w
    echo "Starting ${NUM_WORKERS} workers with name prefix \"${WORKER_NAME}\""
    if ! ${MDCS_BIN}/startworker -name "${WORKER_NAME}" -jobmanagerhost ${JMHOST} -jobmanager "${JOB_MANAGER_NAME}" -num ${NUM_WORKERS} > /dev/null 2>&1
    then
        echo "Could not start workers \"${WORKER_NAME}\""
        exit 4
    fi
}

setupWorkerNode() {

    # Start MDCE service
    cleanMdcsStart
    
    # Start all the workers
    startWorkers

    prewarmMATLAB
}

setupHeadNode() {

    # Start MDCE service
    cleanMdcsStart
    
    # The headnode has the cluster's job manager running on it. Start it up.
    echo "Starting the job manager"
    if ${MDCS_BIN}/startjobmanager -certificate ${CERT_FILE} -name "${JOB_MANAGER_NAME}" -cleanPreserveJobs
    then
        # Start all the workers
        startWorkers
    else
        echo "Could not start job manager \"${JOB_MANAGER_NAME}\""
        exit 3
    fi

    prewarmMATLAB
}

prewarmMATLAB() {
    # Call the prewarmMATLAB script in the background. This reads files into memory
    # to improve MATLAB startup time. This is just an optimization, so there's no 
    # need to block.
    cd ${MLROOT}
    ${MLROOT}/bin/glnxa64/MATLABStartupAccelerator ${MLROOT} ${MLROOT}/toolbox/distcomp/bin/util/msa_pct.ini &
    cd -
}

########################################
#
#   Script starts here
#
########################################

# Echo the username of this script
echo "Starting mdce as $(whoami)."

#Check required environment variables. Echo problems to stderr and exit
if [ -z "${NODE_TYPE}" ]; then
    echo "NODE_TYPE environment variable not set" 1>&2
    exit 1
fi

if [ -z "${MDCS_BIN}" ]; then
    echo "MDCS_BIN environment variable not set" 1>&2
    exit 1
fi

if [ -z "${HEADNODE_NAME}" ]; then
    echo "HEADNODE_NAME environment variable not set" 1>&2
    exit 1
fi

if [ -z "${KEYSTORE_FILE}" ]; then
    echo "KEYSTORE_FILE environment variable not set" 1>&2
    exit 1
fi

# Check the keystore file exists
if [ ! -f "${KEYSTORE_FILE}" ]; then
    echo "KEYSTORE_FILE ${KEYSTORE_FILE} is not a file" 1>&2
    exit 1
fi

if [ -z "${CERT_FILE}" ]; then
    echo "CERT_FILE environment variable not set" 1>&2
    exit 1
fi

# Check the certificate file exists
if [ ! -f "${CERT_FILE}" ]; then
    echo "CERT_FILE ${CERT_FILE} is not a file" 1>&2
    exit 1
fi

# Set the default values for some optional environment variables. 
. ${MDCS_BIN}/util/set_default_environment_values.sh

# Make sure MDCE_TMP exists
mkdir "${MDCE_TMP}"

export TMP="${MDCE_TMP}"

# Use binary connect to help large MPI rings
export MDCE_CONNECTACCEPT=binary

# Log a lot in this script
VERBOSE=1

# Log a lot in MDCS
if [ -z "${MDCE_LOG_LEVEL}" ]; then
    MDCE_LOG_LEVEL=2
fi

METADATA_IP=169.254.169.254

# This will be passed to the mdce command via -hostname
MDCS_PUBLIC_HOSTNAME=`curl --silent http://${METADATA_IP}/latest/meta-data/public-hostname`

# Job manager name 
if [ -z "${JOB_MANAGER_NAME}" ]; then
    JOB_MANAGER_NAME=EC2_job_manager
fi
    
# Avoid letting MPICH2 rely on DNS within EC2. MPICH2 by default uses short 
# DNS names, which are somewhat unreliable. Long names seem to work better,
# but the only diagnosed problem so far is with strange midfixes in EC2
# internal domain names. IP should be better.
# export MPICH_INTERFACE_HOSTNAME=`hostname -f`
export MPICH_INTERFACE_HOSTNAME=`curl --silent http://${METADATA_IP}/latest/meta-data/local-ipv4`

echo "mdce will be started on ${MDCS_PUBLIC_HOSTNAME}"

# Print the environment before starting mdce
if [ -z "${VERBOSE}" ] || [ "${VERBOSE}" -eq 1 ]; then
    echo "-----"
    echo "Environment used to start mdce:"
    echo
    printenv | sort
    echo "-----"
fi

# Act appropriately to the node's type
case ${NODE_TYPE} in
    HEADNODE )
        verbose "Setting up a headnode on ${MDCS_PUBLIC_HOSTNAME}"
        setupHeadNode
        ;;
    WORKER )
        verbose "Setting up a worker machine on ${MDCS_PUBLIC_HOSTNAME}"
        setupWorkerNode
        ;;
    NONE )
        verbose "Doing nothing with MDCS on ${MDCS_PUBLIC_HOSTNAME}"
        ;;
    *)
        echo "Unknown NODE_TYPE request: ${NODE_TYPE}"
        exit 3
        ;;
esac

echo "Finished starting MDCS on ${MDCS_PUBLIC_HOSTNAME}"

exit 0
