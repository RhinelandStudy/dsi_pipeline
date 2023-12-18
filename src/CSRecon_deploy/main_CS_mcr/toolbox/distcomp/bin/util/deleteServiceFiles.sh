#!/bin/sh

# Copyright 2008-2017 The MathWorks, Inc.

#---------------------------------------------
# Delete all files related to the mdce service
#
# The environment variables UTILBASE, CHECKPOINTBASE, LOGBASE, PIDFILE and
# LOCKFILEs need to be defined before calling this script.
#
# MDCEUSER also needs to be defined whenever appropriate.
#---------------------------------------------
executeWithRetry() {
    for i in `seq 1 5`; do
        eval "$1" # Run the command
 
        if [ $? -eq 0 ]; then
            return 0; # If successful, return
        fi

        if [ $i -eq 5 ]; then
            return 1; # Abort after 5 retries
        fi

        sleep 5; # Retry after a pause
    done
}

deleteFilesAsMDCEUser() {
    if [ -n "$MDCEUSER" ]; then
        executeWithRetry "sudo -u $MDCEUSER \"$1\""
        if [ $? -ne 0 ]; then
            echo "Unable to delete files as user $MDCEUSER."
            return 1;
        fi
    else
        executeWithRetry "\"$1\""
        if [ $? -ne 0 ]; then
            echo "Unable to delete files."
            return 1;
        fi
    fi
}

deleteAllServiceFilesOrExit() {
    deleteFilesAsMDCEUser "\"$UTILBASE\"/deleteMDCEUserFiles.sh"
    if [ $? -ne 0 ]; then
        exit 1;
    fi
    
    executeWithRetry "\"$UTILBASE\"/deleteInvokingUserFiles.sh"
    if [ $? -ne 0 ]; then
        echo "Unable to delete files."
        exit 1;
    fi
}

deleteDatabaseFilesOrExit() {
    deleteFilesAsMDCEUser "\"$UTILBASE\"/deleteDatabaseFiles.sh"
    if [ $? -ne 0 ]; then
        exit 1;
    fi
}

deleteSecurityFilesOrExit() {
    deleteFilesAsMDCEUser "\"$UTILBASE\"/deleteSecurityFiles.sh"
    if [ $? -ne 0 ]; then
        exit 1;
    fi
}
