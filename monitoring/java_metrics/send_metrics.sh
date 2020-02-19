#!/usr/bin/env bash
#
#
#  Sends count of dependencies and jar size to monitoring services
#  graphite service for monitoring.
#  Author        :Alexander Stoyakin                                             
#  Email         :alexander@skymind.global
#
#
###############################################################################

HOST=
PORT=
JAR_PATH=

if [ $# -eq 3 ]
then
   HOST=$1
   PORT=$2
   JAR_PATH=$3
else
   echo "Usage: send_metrics.sh HOST PORT JAR_PATH"
fi

mvn dependency:tree > deps.txt

let cnt=0
while read -r line
   do
        data=`echo "$line" | grep -o '+-'`
        if [ ! -z $data ]; then
   	  ((cnt++))
        fi
done < deps.txt

echo "Detected $cnt transitive dependencies."

stat="dl4j.java.metrics.dependencies_count $cnt `date +%s`"
echo $stat | nc -q0 $HOST $PORT

mvn install
size="stat -c%s $JAR_PATH"
stat="dl4j.java.metrics.jar_size $size `date +%s`"
echo $stat | nc -q0 $HOST $PORT



