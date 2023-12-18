#!/bin/sh

# Copyright 2004-2017 The MathWorks, Inc.

printPathMessage() {
    echo "Paths specified in MDCS_ADDITIONAL_MATLABROOTS must refer to"
    echo "the root of an MDCS installation from R2016a or later."
}

printFutureVersionMessage() {
    echo ""
    echo "WARNING: $1"
    echo "refers to a newer MATLAB release ($2) than the current MATLAB"
    echo "release ($3). Only older MATLAB releases can be used as additional"
    echo "installations for MJS. To use MATLAB $2 with this MJS, you must"
    echo "upgrade your MJS to $2 or later."
    echo "Ignoring \"$1\"."
}

# From http://stackoverflow.com/a/8811800/413172
contains() {
    string="$1"
    substring="$2"
    if test "${string#*$substring}" != "$string"
    then
        return 0    # $substring is in $string
    else
        return 1    # $substring is not in $string
    fi
}

checkPCTVersion() {
    MATLABROOT=$1
    # We need to check the PCT version to make sure only previous
    # releases have been specified.
    # The script for getting the PCT version only shipped in
    # R2016b so it's not fatal if we can't find it. It's only used
    # to error if a _future_ version has been specified.
    GET_VERSION_BIN=$MATLABROOT/toolbox/distcomp/bin/util/getPCTVersionNum.sh
    if [ -x "$GET_VERSION_BIN" ]
    then
        PCT_VERSION=$($GET_VERSION_BIN)
        if [ $PCT_VERSION -gt $MDCS_CURRENT_VERSION ]
        then
            printFutureVersionMessage $MATLABROOT $MATLAB_RELEASE $MDCS_CURRENT_RELEASE
            return 1
        else
            return 0
        fi
    fi
}

getMATLABRelease() {
    MATLABROOT=$1
    # Get the release of MATLAB from the given MATLABROOT, first
    # check that pctGetMATLABRelease exists and is executable
    # and warn if not. Otherwise we execute it to get the
    # MATLAB release for this MATLABROOT.
    GET_RELEASE_BIN=$MATLABROOT/bin/$ARCH/pctGetMATLABRelease
    if [ -x "$GET_RELEASE_BIN" ]
    then
        MATLAB_RELEASE=$($GET_RELEASE_BIN)
        return 0
    else
        echo ""
        echo "WARNING: Cannot determine MATLAB release installed at:"
        echo "$MATLABROOT"
        printPathMessage
        return 1
    fi
}

addReleaseIfNotSpecified() {
    # Check if MATLAB_EXECUTABLE already contains an entry for this
    # MATLAB release. We check for "<release>=" to make sure we don't
    # accidentally match a part of the path as the release.
    if contains "$MATLAB_EXECUTABLE" "$MATLAB_RELEASE="
    then
        echo ""
        echo "WARNING: An MDCS installation for $MATLAB_RELEASE has already been specified."
        echo "There must be only one MDCS installation specified for each release."
        echo "Ignoring \"$MATLABROOT\"."
    else
        echo ""
        echo "Found additional MDCS installation for $MATLAB_RELEASE at:"
        echo "$MATLABROOT"
        MATLAB_EXECUTABLE="$MATLAB_EXECUTABLE;$MATLAB_RELEASE=$MATLABROOT/$PATH_TO_MATLAB"
    fi
}

checkAndAddMATLABRelease() {
    MATLABROOT=$1
    if [ -d "$MATLABROOT" ]
    then
        # getMATLABRelease sets the MATLAB_RELEASE variable, or prints a
        # warning and returns an error code if not.
        if getMATLABRelease $MATLABROOT
        then
            # Now check the PCT version. It is possible that a future release 
            # has been specified, this function will print a warning and
            # return an error code in that case.
            if checkPCTVersion $MATLABROOT
            then
                # Now we know we have a compatible release we can add it to
                # MATLAB_EXECUTABLE if it hasn't already been specified.
                addReleaseIfNotSpecified
            fi
        fi
    else
        echo ""
        echo "WARNING: The path \"$MATLABROOT\" does not exist."
        printPathMessage
    fi
}

parseMatlabroots() {
    # Parse all additional matlabroots specified in the mdce_def file and
    # add them to the MATLAB_EXECUTABLE environment variable.
    # This now contains all the paths to the executables supported by this
    # cluster in the form:
    # MATLAB_EXECUTABLE="<release string>=/path/to/executable;<release string>= etc."

    # If we've not defined the environment already we need to use bin/matlab
    # to start MATLAB. This allows us to start MATLABs from different releases
    # with their correct environment. As this might cause unforseen problems
    # the path can be changed back to bin/$ARCH/MATLAB by setting the
    # DEFINE_MATLAB_ENVIRONMENT variable in the 'mdce' script. See g1339520.
    if [ -z "$DEFINE_MATLAB_ENVIRONMENT" ]; then
        PATH_TO_MATLAB="bin/matlab"
    else
        PATH_TO_MATLAB="bin/$ARCH/MATLAB"
    fi

    # The MATLAB release of this MDCS installation is always included in
    # the MATLAB_EXECUTABLE list.
    MDCS_CURRENT_RELEASE=$($MATBASE/bin/$ARCH/pctGetMATLABRelease)
    MDCS_CURRENT_VERSION=$($UTILBASE/getPCTVersionNum.sh)
    MATLAB_EXECUTABLE="$MDCS_CURRENT_RELEASE=$MATBASE/$PATH_TO_MATLAB"
    if [ -n "$MDCS_ADDITIONAL_MATLABROOTS" ]
    then
        MATLABROOTS=$(echo $MDCS_ADDITIONAL_MATLABROOTS | tr ":" "\n")
        for i in $MATLABROOTS
        do
            checkAndAddMATLABRelease $i
        done
        # Add some vertical whitespace to the output for readability
        echo ""
    fi
}


#-----------------------------------------------------------------------------
# Define some general variables about phoenix
#-----------------------------------------------------------------------------
BINBASE="$MDCEBASE/bin/$ARCH"
APPNAME="mdced"
APP_LONG_NAME="MATLAB Distributed Computing Server"
# Don't change this else we will specifically need to remove it in the stop
# command
PIDFILE="$PIDBASE/$APPNAME.pid"
LOCKFILE="$LOCKBASE/$APPNAME"

# Wrapper
WRAPPER_CMD="$MATBASE/bin/$ARCH/$APPNAME"
WRAPPER_CONF="$CONFIGBASE/wrapper-phoenix.config"
MDCE_PLATFORM_WRAPPER_CONF="$CONFIGBASE/wrapper-phoenix-$ARCH.config"

# Set the MATLAB_EXECUTABLE variable.
# If MDCEQE_MATLAB_EXECUTABLE has been set using the -matlabexecutable flag, use
# this variable for MATLAB_EXECUTABLE and assume that it is correctly formatted.
# Otherwise parse the additional matlabroots defined in the mdce_def file and
# warn if they are not valid. Setting MATLAB_EXECUTABLE is only relevant when
# calling "mdce start", so only run this code if the $ACTION is "start" or 
# "console".
if [ "$ACTION" = "start" ] || [ "$ACTION" = "console" ]
then
    if [ -n "$MDCEQE_MATLAB_EXECUTABLE" ]
    then
        MATLAB_EXECUTABLE="$MDCEQE_MATLAB_EXECUTABLE"
    else
        export MDCS_ADDITIONAL_MATLABROOTS
        parseMatlabroots
    fi
fi

#-----------------------------------------------------------------------------
# Export the variables that are REQUIRED by the wrapper-phoenix.config
# file. These variables must be set correctly for the wrapper layer to
# work correctly.
#-----------------------------------------------------------------------------
export JRECMD_FOR_MDCS
export JREFLAGS

export JREBASE
export MATBASE
export JARBASE
export JAREXTBASE
export JAREXTBASEUTIL
export JINILIB

export MDCE_DEFFILE
export MDCEBASE
export LOGBASE
export CHECKPOINTBASE

export HOSTNAME
export ARCH

export WORKER_START_TIMEOUT

export MATLAB_EXECUTABLE

export JOB_MANAGER_MAXIMUM_MEMORY
export MDCEQE_JOBMANAGER_DEBUG_PORT
export CONFIGBASE

export DEFAULT_JOB_MANAGER_NAME
export DEFAULT_WORKER_NAME

export JOB_MANAGER_HOST
export BASE_PORT

export LOG_LEVEL

export MDCE_PLATFORM_WRAPPER_CONF

export WORKER_DOMAIN
export SECURITY_LEVEL
export USE_SECURE_COMMUNICATION
export TRUSTED_CLIENTS
export SHARED_SECRET_FILE
export SECURITY_DIR
export DEFAULT_KEYSTORE_PATH
export KEYSTORE_PASSWORD
export MDCE_ALLOW_GLOBAL_PASSWORDLESS_LOGON
export ALLOW_CLIENT_PASSWORD_CACHE
export ADMIN_USER
export ALLOWED_USERS

export RELEASE_LICENSE_WHEN_IDLE

export MDCS_ALL_SERVER_SOCKETS_IN_CLUSTER
export MDCS_DUPLEX_PEER_RMI
export MDCS_JOBMANAGER_PEERSESSION_MIN_PORT
export MDCS_JOBMANAGER_PEERSESSION_MAX_PORT
export MDCS_WORKER_MATLABPOOL_MIN_PORT
export MDCS_WORKER_MATLABPOOL_MAX_PORT

export MDCS_WORKER_PROXIES_POOL_CONNECTIONS

export MDCS_LIFECYCLE_REPORTER
export MDCS_LIFECYCLE_WORKER_HEARTBEAT
export MDCS_LIFECYCLE_TASK_HEARTBEAT

export MDCS_PEER_LOOKUP_SERVICE_ENABLED

export MDCS_ADDITIONAL_CLASSPATH

export MDCS_REQUIRE_WEB_LICENSING

export MDCS_SEND_ACTIVITY_NOTIFICATIONS
export MDCS_SCRIPT_ROOT

export MDCS_PEERSESSION_KEEP_ALIVE_PERIOD
export MDCS_PEERSESSION_KEEP_ALIVE_TIME_UNIT

export MDCS_MATLAB_DRIVE_ENABLED_ON_WORKER
export MW_MATLAB_DRIVE_FOLDER_LOCATION_CFG

export MDCS_ALLOW_RESIZING
