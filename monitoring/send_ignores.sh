#!/usr/bin/env bash
#
#
#  Sends ignored tests count to  graphite service for monitoring.
#  Author        :Alexander Stoyakin                                             
#  Email         :alexander@skymind.global
#
#  Example: send_ignores.sh deeplearning4j/nd4j 127.0.0.1 111
#  Arguments: 
#  1. Source code directory to search for ignored tests in.
#  2. Graphite host
#  3. Graphite port
#
###############################################################################

main() {
  declare code_path=$1
  declare HOST=$2
  declare PORT=$3

  ../java-testing/ignores/parse-for-ignores.sh $code_path > output.txt

  ignored=`grep -c IGNORE output.txt`

  stat="nd4j.techdebt.ignore-count $ignored `date +%s`"
  echo $stat
  echo $stat | nc -q0 $HOST $PORT
}

main "$@"
