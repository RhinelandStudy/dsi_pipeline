#!/bin/sh
#
# Copyright 2014 The MathWorks, Inc.
#
# This script is intended to be dot-sourced from any script that is run at 
# cluster startup to set default values for the environment variables. 
# 
# This has been fatored out of start_mdce_on_ec2.sh to allow these default 
# values to be shared between start_mdce_on_ec2.sh and mount_gds.sh
# 
# See the documentation on start_mdce_on_ec2.sh for details of the meaning of 
# these environment variables. 

# Set TMP to a place on the big ephemeral space so that each worker can 
# take advantage of this large area without accidental crosstalk.

if [ -z "${MDCE_TMP}" ]; then
    MDCE_TMP=/shared/tmp/mdceworkertmp
fi

if [ -z "${MDCE_EPHEMERAL_BASE}" ]; then
    MDCE_EPHEMERAL_BASE=/mnt/mdce
fi

if [ -z "${MDCE_LOG_PATH}" ]; then
    MDCE_LOG_PATH="${MDCE_EPHEMERAL_BASE}"/log
fi

