#!/usr/bin/env bash

# Copyright 2013-2015 The MathWorks, Inc.

## ulimit is a shell built-in. It has different options in bash and dash.
## We specify in the shebang to use bash explicitly so that we always get
## the right answer. See g967131.

printNprocWarningUlimit() {
    echo ""
    echo "WARNING: The mdce script detected that the number of processes allowed"
    echo "is limited by ulimit. Be sure that the limit of processes for the ROOT user"
    echo "(or the user running the mdce service) is set to either \"unlimited\""
    echo "or at least 128 * W, where W is the maximum number of MATLAB Distributed"
    echo "Computing Server workers that will run on this machine."
    echo ""
}

# We are using the bash-specific -u flag which gives us the maximum
# number of of processes available to a single user. See the bash man
# page.
numOfLimits=`ulimit -u`
if [ "$numOfLimits" != "unlimited" ]; then
    printNprocWarningUlimit
fi
