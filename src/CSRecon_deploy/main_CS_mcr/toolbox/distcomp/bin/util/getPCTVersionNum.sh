#!/bin/sh

# Copyright 2016 The MathWorks, Inc.

#========================= realpath.sh (start) ============================
#-----------------------------------------------------------------------------
# Usage: realpath <filename>
# Returns the actual path in the file system of a file. It follows links. 
# It returns an empty path if an error occurs.
# Return status: 0 if successful.
# If return status is 0, the function echoes out the real path to the file.
# Return status 1 Exceeded the maximum number of links to follow.
# Return status 2 Some other error occurred.
realpath() {
    filename_rpath=$1
    SUCCESS_STATUS_rpath=0
    MAX_LINKS_EXCEEDED_rpath=1
    OTHER_ERROR_rpath=2
    #
    # Now filename_rpath is either a file or a link to a file.
    #
    cpath_rpath=`pwd`

    #
    # Follow up to 8 links before giving up. Same as BSD 4.3
    # We cd into the directory where the file is located, and do a /bin/pwd 
    # to get the name of the CWD.  If the file is a symbolic link, we update 
    # the basename of the file and cd into the directory that the link points 
    # to and repeat the process.
    # Once we arrive in a directory where we do not have a soft-link, we are 
    # done.
    n_rpath=1
    maxlinks_rpath=8
    while [ $n_rpath -le $maxlinks_rpath ]
    do
        #
        # Get directory part of $filename_rpath correctly!
        #
        newdir_rpath=`dirname "$filename_rpath"`
        # dirname shouldn't return empty instead of ".", but let's be paranoid.
        if [ -z "${newdir_rpath}" ]; then
            newdir_rpath=".";
        fi
        (cd "$newdir_rpath") > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            # This should not happen.  The file is in a non-existing directory.
            cd "$cpath_rpath"
            return $OTHER_ERROR_rpath
        fi
        cd "$newdir_rpath"
        #
        # Need the function pwd - not the shell built-in one.  The command 
        # /bin/pwd resolves all symbolic links, but the shell built-in one 
        # does not.
        #
        newdir_rpath=`/bin/pwd`
        # Stip the directories off the filename_rpath.
        newbase_rpath=`basename "$filename_rpath"`

        lscmd=`ls -l "$newbase_rpath" 2>/dev/null`
        if [ ! "$lscmd" ]; then
            # This should not happen, the file does not exist.
            cd "$cpath_rpath"
            return $OTHER_ERROR_rpath
        fi
        #
        # Check for link.  The link target is everything after ' -> ' in 
        # the output of ls.
        #
        if [ `expr "$lscmd" : '.*->.*'` -ne 0 ]; then
            filename_rpath=`echo "$lscmd" | sed 's/.*-> //'`
        else
            #
            # We are done.  We found a file and not a symbolic link.
            # newdir_rpath contains the directory name, newbase_rpath contains 
            # the file name.
            cd "$cpath_rpath"
            echo "$newdir_rpath/$newbase_rpath"
            return $SUCCESS_STATUS_rpath
        fi
        n_rpath=`expr $n_rpath + 1`
    done
    # We exceeded the maximum number of links to follow.
    cd "$cpath_rpath"
    return $MAX_LINKS_EXCEEDED_rpath
}
#========================= realpath.sh (end) ==============================

#========================= pathsetup.sh (start) ============================
#-----------------------------------------------------------------------------
# Usage: warnIfNotInBinUtil <full path> <scriptname>
warnIfNotInBinUtil() {
    # Search for toolbox/distcomp/bin/util in $1.
    if [ `expr "$1" : ".*toolbox/distcomp/bin/util$"` -eq 0 ]; then
        echo "Warning: $2 should be run only from toolbox/distcomp/bin/util,"
        echo "or using a symbolic link to toolbox/distcomp/bin/util/$2."
        echo ""
    fi
}

# THIS MUST BE RUN FIRST BEFORE SETTING UP THE APPLICATION VARIABLES
# Get the fully qualified path to the script
SCRIPT="$0"
REALPATH=`realpath "$SCRIPT"`
# Get the path to distcomp bin BASE from this by removing the name of the shell
# script.
BASE=`echo $REALPATH | sed -e 's;\/[^\/]*$;;g'`
warnIfNotInBinUtil "$BASE" util/getPCTVersionNum.sh
# Make sure we are in the correct directory to run setbase.sh
cd "$BASE"
# We're in the bin/util directory, so go up one directory to get to bin
cd ..
# Set base directory variables
. util/setbase.sh
cd "$BASE"
#-----------------------------------------------------------------------------
#========================= pathsetup.sh (end) ==============================

. "$UTILBASE/setmdceenv"
defineJRECMD

$JRECMD \
    -classpath "$DISTCOMP_ONLY_CLASSPATH" \
    com.mathworks.toolbox.distcomp.util.Version -num
