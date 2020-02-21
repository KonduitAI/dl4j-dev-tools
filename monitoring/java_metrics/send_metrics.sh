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

HOST="51.107.91.115"
PORT="2003"

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
size='stat -c%s target/deeplearning4j-monitoring-1.0.0-beta6.jar'
stat="dl4j.java.metrics.jar_size $size `date +%s`"
echo $stat | nc -q0 $HOST $PORT



