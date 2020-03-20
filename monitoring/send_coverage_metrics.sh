#!/usr/bin/env bash
#
#
#  Parses nd4j application.log file and sends extracted coverage percentage to
#  graphite service for monitoring.
#  Author        :Alexander Stoyakin                                             
#  Email         :alexander@skymind.global
#
#
#  Usage example: ./send_coverage_metrics.sh ~/deeplearning4j/nd4j/nd4j-backends/nd4j-tests/logs/application.log 127.0.0.1 1111
#
#  Positional arguments:
#  1. log file path
#  2. graphite host
#  3. graphite port for metrics
#
###############################################################################

main() {
  declare log_path=$1
  declare HOST=$2
  declare PORT=$3

  echo -e "\nINFO:Running checks on: ${log_path}\n"

  if [ -z $HOST ] || [ -z $PORT ]; then
          echo "Usage: send_coverage_metrics.sh log_file monitoring_host monitoring_port"
          exit 6
  fi

  if [[ ! -e "$log_path" ]]; then
    echo -n "ERROR: $log_path DOES NOT exist." 1>&2
    echo -n "Supplied path has to be absolute" 1>&2
    echo "Exiting script..." 1>&2
    exit 7
  fi


   category=""
   while read -r line
   do
        data=`echo "$line" | grep -o '[^ ]*%' | sed "s/[^0-9.,]//g"`
        if [ ! -z $data ];then
                if [ -z $category ]; then
                  category='forward'
                elif [ $category = "forward" ]; then
                        category='backward'
                elif [ $category = "backward" ]; then
                        category="tf_mapped"
                elif [ $category = "tf_mapped" ]; then
                        category="tf_mapped_tested"
                elif [ $category = "tf_mapped_tested" ]; then
                        category="libnd4j_mapped"
                fi
                stat="nd4j.techdebt.op-coverage.$category $data `date +%s`"
                echo $stat
                echo $stat | nc -q0 $HOST $PORT
        fi
   done < "$log_path"
}

main "$@"
