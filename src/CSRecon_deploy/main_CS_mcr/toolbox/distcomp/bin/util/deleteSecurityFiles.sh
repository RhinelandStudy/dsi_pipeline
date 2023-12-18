#!/bin/sh

# Copyright 2013 The MathWorks, Inc.

#---------------------------------------------
# Delete the files related to the jobmanager security that are owned by either the
# invoking user or MDCEUSER.  This script should be invoked via sudo whenever
# MDCEUSER is defined.
#
# The environment variable SECURITY_DIR needs to be defined before calling this script.
#---------------------------------------------

# Make sure we don't accidentally delete any files by checking whether
# environment variables are defined.
# Exit immediately upon failure.

if [ -n "$SECURITY_DIR" ]; then
    rm -f "$SECURITY_DIR"/aes_private         || exit 1
    rm -f "$SECURITY_DIR"/private             || exit 1
    rm -f "$SECURITY_DIR"/public              || exit 1
fi
