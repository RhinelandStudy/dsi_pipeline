#!/bin/sh
<HEADERS>

# Copyright 2006-2012 The MathWorks, Inc.

# Check that our environment is intact
if [ ${MDCE_DECODE_FUNCTION:-X} = X ] ; then
    echo "Fatal error: environment variable MDCE_DECODE_FUNCTION is not set on the cluster"
    echo "This may happen if you have used '-v' in your scheduler SubmitArguments"
    echo "Please either use another means to transmit the information, or use '-V'"
    exit 1
fi

# Put ourselves in the TMPDIR for job execution - create one if none exists.
if [ ${TMPDIR:-X} = X ] ; then
    # Create a job directory
    TMPDIR=/tmp/${PBS_JOBID?"PBS_JOBID not defined!"}
    mkdir -p ${TMPDIR}
    export TMPDIR
    echo "Created directory: ${TMPDIR} on `hostname`"
    trap "cd /tmp ; rm -rf ${TMPDIR} ; echo Removed ${TMPDIR}" 0 1 2 15
fi
cd ${TMPDIR}
