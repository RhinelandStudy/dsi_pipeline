#!/bin/sh
#=======================================================================
# Functions:  Required by call to arch.sh
#   check_archlist ()
#=======================================================================

# Copyright 2004-2016 The MathWorks, Inc.

 check_archlist () {
         return 0
 }

 # Resolve the location of the 'ps' command
 PSEXE="/usr/bin/ps"
 if [ ! -x $PSEXE ]
 then
     PSEXE="/bin/ps"
     if [ ! -x $PSEXE ]
     then
         echo "Unable to locate 'ps'."
         exit 1
     fi
fi
# All scripts should call this function after they have changed to the
# MATLABROOT/toolbox/distcomp/bin directory
# Reinitialise the BASE variable to pick up any '..' type elements in the call
BASE=`pwd`
# Define the MATBASE (MATLABROOT) directory
MATBASE=`echo $BASE | sed -e 's;/toolbox/distcomp/bin;;g'`
MDCEBASE="$MATBASE/toolbox/distcomp"
CONFIGBASE="$MDCEBASE/config"
UTILBASE="$MDCEBASE/bin/util"

JARBASE="$MATBASE/java/jar/toolbox"
JAREXTBASEUTIL="$MATBASE/java/jarext"
JAREXTBASE="$MATBASE/java/jarext/distcomp"
JINILIB="$JAREXTBASE/jini2/lib"

MATJARBASE="${MATBASE}/java/jar"

MAT_UTIL_JAR="${MATJARBASE}/util.jar"
MAT_FOUNDATION_JAR="${MATJARBASE}/foundation_libraries.jar"
DISTCOMP_REQS="${JAREXTBASEUTIL}/commons-lang.jar"
MAT_RESOURCE_CORE="${MATJARBASE}/resource_core.jar"
MAT_PARALLEL_RES="${MATJARBASE}/resources/parallel_res.jar"
WEB_CLIENTS_CORE_JAR="${JAREXTBASEUTIL}/webservices/ws_client_core/mw-service-client-core.jar"
GDS_JOBS_CLIENT_JAR="${JAREXTBASEUTIL}/webservices/gds_jobs_client/gds-jobs-client.jar"
MLWEBSERVICES_JAR="${MATJARBASE}/mlwebservices.jar"
WEBPROXY_JAR="${MATJARBASE}/webproxy.jar"
NET_JAR="${MATJARBASE}/net.jar"
JSON_JAR="${JAREXTBASEUTIL}/gson.jar"

# The jars required to make remote commands to start/stop services
REMOTE_COMMAND_REQS="$JINILIB/start.jar:$JINILIB/destroy.jar:$JINILIB/phoenix.jar:$JINILIB/reggie.jar:$JINILIB/jini-ext.jar"
# The jars required to use remote execution
REMOTE_EXECUTION_REQS="$MATBASE/java/jarext/jsch.jar"
# The jars required to run code on the client (cf. job manager admin registration)
REMOTE_CLIENT_REQS="$MATBASE/java/jar/jmi.jar:$MATBASE/java/jar/mvm.jar:$MATBASE/java/jar/services.jar"

DISTCOMP_ONLY_CLASSPATH="${JARBASE}/distcomp.jar:${JARBASE}/parallel/pctutil.jar:${JARBASE}/parallel/util.jar"
# The classpath that all the start and stop scripts should use.
REMOTE_COMMAND_CLASSPATH="$DISTCOMP_ONLY_CLASSPATH:$REMOTE_COMMAND_REQS:${MAT_UTIL_JAR}:${DISTCOMP_REQS}:${MAT_FOUNDATION_JAR}:${MAT_RESOURCE_CORE}:${MAT_PARALLEL_RES}:${WEB_CLIENTS_CORE_JAR}:${GDS_JOBS_CLIENT_JAR}:${MLWEBSERVICES_JAR}:${WEBPROXY_JAR}:${NET_JAR}:${JSON_JAR}:${REMOTE_CLIENT_REQS}"
# The classpath that all the remote scripts should use
REMOTE_EXECUTION_CLASSPATH="$DISTCOMP_ONLY_CLASSPATH:$REMOTE_EXECUTION_REQS:${MAT_UTIL_JAR}:${DISTCOMP_REQS}:${MAT_FOUNDATION_JAR}:${MAT_RESOURCE_CORE}:${MAT_PARALLEL_RES}:${WEB_CLIENTS_CORE_JAR}"


# Call arch.sh to define the $ARCH variable - this is the same as is
# used in setmlenv
ARCH=""
. "$MATBASE/bin/util/arch.sh"

# Library path for the java
NATIVE_LIBRARY_PATH="${MATBASE}/bin/${ARCH}"
