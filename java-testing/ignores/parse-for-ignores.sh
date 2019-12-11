#!/usr/bin/env bash
#
# Finds ignored test classes and test methods in the given project directory
# Main method is named parse_for_ignores
#	Arguments:
#		1) Absolute path to run checks in. Checks files under src/test
#
#	Usage examples: ./parse-for-ignores ~/deeplearning4j/nd4j
#				    ./parse-for-ignores ~/deeplearningdatavec/datavec-spark
#
#   Globals:
#		LIST_OF_TEST_DIRS
###############################################################################

set -o errexit
set -o pipefail

###############################################################################
# Checks if supplied test class is ingored in entirety
# Captures comment inside @Ignore(...) if present
# Arguments:
#   1) Java test file name
###############################################################################
ignored_all() {
  sed 's/\/\/.*//' < "$1" \
   | perl -0p -e \
       's/\/\*(.*?)\*\///sg;' -e \
       's/\@Ignore(\((\".*?\")\))*.*?public\s+class\s+(\S+)/\nFLAG,$2\n/sg' \
   | grep ^FLAG \
   | sed 's/FLAG/*/g'
}

###############################################################################
# Finds ignored tests in given test file
# Captures comment inside @Ignore(...) if present
# Arguments:
#   1) Java test file name
###############################################################################
ignored_tests() {
  sed 's/\/\/.*//' < "$1" \
   | perl -0p -e \
       's/\/\*(.*?)\*\///sg;' -e \
       's/\@Ignore(\((\".*?\")\))*.*?public\s+void\s+(\S+)\(/\nFLAG$3,$2\n/sg' \
   | grep ^FLAG \
   | perl -pe 's/FLAG//g'
}

############################################################################### 
# Finds ignored tests in given test file                                        
# Captures comment inside @Ignore(...) if present                               
# Arguments:                                                                    
#   1) Java test file name                                                      
############################################################################### 
#ignored_tests_opvalidation_suite() {                                                               
# perl -pe 's/(^\s*\/\/|\s+\/\/).*$//' < "$1" \                                  
#   | tr -d "\n" \                                                               
#   | perl -pe \                                                                 
#	 's/\@Ignore(\((\".*?\")\))*.*?public\s+void\s+(\S+)\(/\nFLAG$3,$2\n/g' \   
#	 's/public\s+void\s+(\S+)\(.*?OpValidationSuite.ignoreFailing();/\nFLAG$3,$2\n/g' \   
#   | grep ^FLAG \                                                               
#   | perl -pe 's/FLAG//g'                                                       
#}  

###############################################################################
# Parse every java file under any src/test and collect information on tests 
# Arguments:
#  None
###############################################################################
parse_test_dirs() {
  #header for csv
  echo -n "IGNORED/ALL IGNORED, "
  echo -n "PACKAGE NAME, TEST CLASS, TEST METHOD,"
  echo "IGNORED COMMENT, LOCAL PATH TO TEST CLASS"
 
  for test_dir in ${LIST_OF_TEST_DIRS[@]}; do

	echo -e "\nINFO:Running in subdir $test_dir..."
    list_of_test_files=$(find "$test_dir" -type f -name "*.java")
   
    for test_file in ${list_of_test_files[@]}; do

      echo "INFO:Checking test file $test_file..."
	  test_file_name=$(basename "$test_file")
      test_class=${test_file_name%.java}
      package_name=$(grep "package .*;" "$test_file" \
	  				  | awk '{print $2}' \
	  				  | tr -d ';')
      #Check if entire test class is ignored, if ignored continue to next 
      all_ignored=$(ignored_all "$test_file") || true
      if [[ -n "$all_ignored" ]]; then
        echo -n "ALL IGNORED, "
        echo "$package_name, $test_class, $all_ignored, $test_file"
        continue
      fi

      #Check for ignored tests in test class
      tests_ignored=$(ignored_tests "$test_file") || true
      if [[ -n "$tests_ignored" ]]; then       
        while read -r line; do
            echo -n "IGNORED, "
			echo "$package_name, $test_class, $line, $test_file"
        done <<< "$tests_ignored"
      fi
	  echo "INFO:Checked test file $test_file..."
    done
    echo "INFO:Finished running subdir $test_dir"

  done
}

###############################################################################
# Main method
# Arguments:
#  1) Absolute path to directory to run checks in
#
###############################################################################
parse_for_ignore() {
  declare test_path=$1
  echo -e "\nINFO:Running checks on: ${test_path}\n"

  if [[ ! -d "$test_path" ]]; then
    echo -n "ERROR: $test_path DOES NOT exist.Supplied path has to be absolute"
    echo "Exiting script..." 1>&2
    exit 7
  else
    LIST_OF_TEST_DIRS=$(find "$test_path" -path "*/src/test")
  fi
  
  if [[ -z "$LIST_OF_TEST_DIRS" ]]; then
    echo "ERROR: NO src/test directories under $test_path" 1>&2
    exit 8
  fi

  parse_test_dirs
  echo -e "\n\nINFO: Run completed."
  exit 0
}

###############################################################################
#Call to main method..
parse_for_ignore "$@"

