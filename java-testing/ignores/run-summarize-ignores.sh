#!/usr/bin/env bash                                                             
#                                                                               
# Finds ignored test classes, test methods and failing tests set to ignore 
# in given project directory. 
#                                                                               
# Author       	:Susan Eraly
# Email        	:susan@skymind.global
#     
# Globals:                                                                    
# PROJECT_DIR                                                             
# RESULTS_DIR

# Note: 
# This is mostly a wrapper around parse_for_ignores.sh that checks arguments and
# separates output to write to different files
###############################################################################

set -o errexit
set -o pipefail

usage=\
'
Usage is:
  run-summarize-ignores.sh <Relative Path to check> <Path for results> [DL4J repo path]
  
  If the optional third argument in not provided, script will check the parent dir
  of the dl4j-dev-tools repo for the deeplearning4j repo.
  
  Example usage:
  run-summarize-ignores.sh nd4j results
  run-summarize-ignores.sh datavec/datavec-spark results_only_datavec_spark
  run-summarize-ignores.sh nd4j results path_to_my_dl4j_repo
'

if [[ "$#" -lt 2 ]]; then
  echo "ERROR: Illegal number of arguments." 1>&2
  echo "$usage" 1>&2
  echo "Exiting script..." 1>&2
  exit 77
fi

declare PROJECT_DIR=$1                                                      
declare RESULTS_DIR="$2"/"$PROJECT_DIR"

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

if [[ $# -eq 3 ]]; then
  if [[ ! -d "$3" ]]; then                                              
    echo -n "ERROR: Provided path to deeplearning4j repo is incorrect." 1>&2
    echo "$3 DOES NOT exist. Provide valid path." 1>&2
    echo "$usage" 1>&2
    echo "Exiting script..." 1>&2   
    exit 7                                                                      
  else   
    DL4J_BASE_DIR="$3"	
  fi
else
  #awk because on mac shell might print "...saving history..." before exiting
  DL4J_BASE_DIR="$(echo $(cd ../../../deeplearning4j && pwd) | awk '{print $1}')"
  echo "INFO: Constructing dl4j repo path based on absolute path of script ..."
  echo "INFO: Checking for repo at $DL4J_BASE_DIR"
  if [[ ! -d "$DL4J_BASE_DIR" ]]; then
    echo -n "ERROR: Cannot find deeplearning4j repo under the parent dir " 1>&2
	echo -n "of the dl4j-dev-tools repo." 1>&2 
    echo -n "Provide a third argument with the absolute path to the repo." 1>&2
	echo "$usage" 1>&2
    echo "Exiting script..." 1>&2
    exit 13
  else
	echo "INFO: Found repo ..."
  fi
fi
                                                                               
TEST_PATH="${DL4J_BASE_DIR}/${PROJECT_DIR}" 
if [[ ! -d "$TEST_PATH" ]]; then                                              
  echo -n "ERROR: Constructed $TEST_PATH DOES NOT exist. " 1>&2
  echo -n "Supplied paths have to be relative to the deeplearning4j repo" 1>&2 
  echo ", i.e the parent dir of libnd4j,nd4j etc" 1>&2
  echo "$usage" 1>&2
  echo "Exiting script..." 1>&2
  exit 7                                                                      
fi

log_file="$RESULTS_DIR"/summarize-ignores.log
info_classes="$RESULTS_DIR"/ignored_classes.csv
info_methods="$RESULTS_DIR"/ignored_methods.csv
info_failed="$RESULTS_DIR"/ignore_failing_methods.csv

./parse-for-ignores.sh "$TEST_PATH" 2>&1 | tee $log_file

echo -n "===================================================================="
echo "===================================================================="

grep "ALL IGNORED" $log_file> $info_classes
line_count=$(wc -l "$info_classes" | awk '{print $1}')
#count can return zero
let "count = $line_count - 1" || true
echo "Ignored classes count:$count. Written to: $info_classes"

grep ^IGNORED $log_file > $info_methods
line_count=$(wc -l "$info_methods" | awk '{print $1}')
let "count = $line_count - 1" || true
echo "Ignored methods count: $count. Written to: $info_methods"

grep "IGNORE FAILING" $log_file > $info_failed
line_count=$(wc -l "$info_failed" | awk '{print $1}')
let "count = $line_count - 1" || true
echo "Ignore failing tests count:$count. Written to: $info_failed"

echo -e "\n\tLog file written to $log_file"
echo -n "===================================================================="
echo "===================================================================="

