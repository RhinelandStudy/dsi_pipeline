#!/bin/sh

# Copyright 2013 The MathWorks, Inc.

#---------------------------------------------
# Delete the files related to the jobmanager database that are owned by either the
# invoking user or MDCEUSER.  This script should be invoked via sudo whenever
# MDCEUSER is defined.
#
# The environment variable CHECKPOINTBASE needs to be defined before calling this script.
#---------------------------------------------

# Make sure we don't accidentally delete any files by checking whether
# environment variables are defined.
# Exit immediately upon failure.

if [ -n "$CHECKPOINTBASE" ] ; then 
    # Directories must not end with a forward slash in order for this to work on
    # sol64.
    rm -rf "$CHECKPOINTBASE"/*_jobmanager_storage || exit 1
fi
