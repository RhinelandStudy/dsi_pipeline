#!/bin/sh
#
# Copyright 2011 The MathWorks, Inc.
#
# Waits, up to a specified number of seconds, for all processes with the 
# specified name to end.
# 
# waitForProcessesWithNameToEnd.sh process_name timeout_seconds
# 
# Example:
#   Wait 5 minutes for all process named "session_helper" to end.
#   $> waitForProcessesWithNameToEnd.sh session_helper 300
#

if [ $# -ne 2 ]; then
    echo "Usage: `basename $0` process_name timeout_seconds"
    exit 1
fi

processName=${1}

maxWaitSecs=${2}

# Get the current time as a unix date string
unixTime() {
    echo $(date +%s)
}

# Figure out when we want to stop waiting.
endTime=$(expr $(unixTime) + ${maxWaitSecs})

# Get the list of pids.
pidList=$(ps -C ${processName} -o pid --no-headers)

# Wait for all the processes to end.
for pid in ${pidList}; do
    while [ -d "/proc/${pid}" -a $(unixTime) -lt ${endTime} ]; do
        sleep 1
    done
done

