
#pip install tf2onnx

cd $DL4J_TEST_RESOURCES/src/main/resources/tf_graphs/examples

find . -name "frozen_model.pb" -print | while read f;do
	set curr = pwd
	cd `dirname $DL4J_TEST_RESOURCES/src/main/resources/tf_graphs/examples/$f`
	output_name=`ls *.prediction.csv 2>/dev/null`
	stripped_output_name=`echo $output_name | cut -d"." -f1`
	echo $stripped_output_name
	cd $curr
	if [ ! -z $stripped_output_name ]
        then
	  python3 -m tf2onnx.convert --input $f --inputs in:0  --outputs $stripped_output_name:0 --output $f.onnx 2>/dev/null
	fi
done

