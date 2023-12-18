#!/bin/sh
# Simple wrapper to re-write exit codes >= 127

# Copyright 2007 The MathWorks, Inc.

# execute arguments
"${@}"

# Re-write exit status
exitCode=$?
if [ ${exitCode} -gt 127 ] ; then
    exitCode=127
fi
exit ${exitCode}
