
# BERT Fine Tuned Model Test Case Generation

This document: Outlines how to reproduce a test case on BERT: pretrained model -> finetuned on MSPR dataset.

Starting point: https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks

## Step 1: Download Data
Script as linked in BERT repository: https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
This script has been copied (unmodified other than noting original source) to model_zoo/bert/download_glue_data.py

Download and extract the msi (see the gist)
```
https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi
```
Extract to: ```TF_Graphs/glue/MSRParaphraseCorpus```

Execute using the following commands: (customize data directory if required)
```
python /TFOpTests/model_zoo/bert/download_glue_data.py --data_dir /TF_Graphs/glue --path_to_mrpc /TF_Graphs/glue/MSRParaphraseCorpus --tasks MRPC
```

## Step 2: Download pretrained model

These are available here: https://github.com/google-research/bert#pre-trained-models
Download BERT-Base, Uncased: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
Extract to directory (```/TF_Graphs/BERT_uncased_L-12_H-768_A-12``` is used here)

## Step 3: Clone BERT repository and run training

```git clone https://github.com/google-research/bert.git```

In these instructions, it is assumed that the BERT directory is cloned into ```/bert```

It is possible to use the TFOpTests docker container (see docker directory in this project):
```
docker run -v C:\DL4J\Git\TFOpTests:/TFOpTests -v C:/DL4J/Git/dl4j-test-resources:/dl4j-test-resources -v C:/Temp/TF_Graphs/:/TF_Graphs/ -v C:/DL4J/Git/bert:/bert/ -it tfops:latest
```

Followed by:
https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks

**Note:** Using a small batch size (4) to reduce memory requirements for training and inference.
A docker container with 12GB allocated should be sufficient for the training procedure.

```
cd /bert
export BERT_BASE_DIR=/TF_Graphs/BERT_uncased_L-12_H-768_A-12
export GLUE_DIR=/TF_Graphs/glue

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/TF_Graphs/mrpc_output/
```

Upon completion, the output will be placed in the specified output directory (```/TF_Graphs/mrpc_output/``` in the code above).

For CPU training (8 core 5960x), the configuration above completed training in approximately 90 minutes.

```
INFO:tensorflow:Saving dict for global step 2751: eval_accuracy = 0.8627451, eval_loss = 0.7639573, global_step = 2751, loss = 0.7639573
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 2751: /TF_Graphs/mrpc_output/model.ckpt-2751
INFO:tensorflow:evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.8627451
INFO:tensorflow:  eval_loss = 0.7639573
INFO:tensorflow:  global_step = 2751
INFO:tensorflow:  loss = 0.7639573
```

## Step 4: Run inference

For inference, data needs to be in a very specific format.

Create a new directory (```mrpc_inference``` is used here)
Add test.tsv to mrpc_inference directory with following content (following glue / MRPC test.tsv format)
```
index	#1 ID	#2 ID	#1 String	#2 String
9	938878	938896	The broader Standard & Poor's 500 Index <.SPX> was 0.46 points lower, or 0.05 percent, at 997.02.	The technology-laced Nasdaq Composite Index .IXIC was up 7.42 points, or 0.45 percent, at 1,653.44.
163	1819056	1819124	Shares in BA were down 1.5 percent at 168 pence by 1420 GMT, off a low of 164p, in a slightly stronger overall London market.	Shares in BA were down three percent at 165-1/4 pence by 0933 GMT, off a low of 164 pence, in a stronger market.
```

Then, perform inference using the following code:
```
export BERT_BASE_DIR=/TF_Graphs/BERT_uncased_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/TF_Graphs/mrpc_output/

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=/TF_Graphs/mrpc_inference \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/TF_Graphs/mrpc_inference/
```



The output probabilities will be written to the output directory:

test_results.tsv
```
0.99860954	0.0013904407
0.0005442508	0.99945575
```

**NOTE: tokenization information is displayed on the screen during inference:**
```
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: test-1
INFO:tensorflow:tokens: [CLS] the broader standard & poor ' s 500 index < . sp ##x > was 0 . 46 points lower , or 0 . 05 percent , at 99 ##7 . 02 . [SEP] the technology - laced nas ##da ##q composite index . ix ##ic was up 7 . 42 points , or 0 . 45 percent , at 1 , 65 ##3 . 44 . [SEP]
INFO:tensorflow:input_ids: 101 1996 12368 3115 1004 3532 1005 1055 3156 5950 1026 1012 11867 2595 1028 2001 1014 1012 4805 2685 2896 1010 2030 1014 1012 5709 3867 1010 2012 5585 2581 1012 6185 1012 102 1996 2974 1011 17958 17235 2850 4160 12490 5950 1012 11814 2594 2001 2039 1021 1012 4413 2685 1010 2030 1014 1012 3429 3867 1010 2012 1015 1010 3515 2509 1012 4008 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 0 (id = 0)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: test-2
INFO:tensorflow:tokens: [CLS] shares in ba were down 1 . 5 percent at 168 pen ##ce by 142 ##0 gm ##t , off a low of 164 ##p , in a slightly stronger overall london market . [SEP] shares in ba were down three percent at 165 - 1 / 4 pen ##ce by 09 ##33 gm ##t , off a low of 164 pen ##ce , in a stronger market . [SEP]
INFO:tensorflow:input_ids: 101 6661 1999 8670 2020 2091 1015 1012 1019 3867 2012 16923 7279 3401 2011 16087 2692 13938 2102 1010 2125 1037 2659 1997 17943 2361 1010 1999 1037 3621 6428 3452 2414 3006 1012 102 6661 1999 8670 2020 2091 2093 3867 2012 13913 1011 1015 1013 1018 7279 3401 2011 5641 22394 13938 2102 1010 2125 1037 2659 1997 17943 7279 3401 1010 1999 1037 6428 3006 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 0 (id = 0)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: test-3
INFO:tensorflow:tokens: [CLS] last year , com ##cast signed 1 . 5 million new digital cable subscribers . [SEP] com ##cast has about 21 . 3 million cable subscribers , many in the largest u . s . cities . [SEP]
INFO:tensorflow:input_ids: 101 2197 2095 1010 4012 10526 2772 1015 1012 1019 2454 2047 3617 5830 17073 1012 102 4012 10526 2038 2055 2538 1012 1017 2454 5830 17073 1010 2116 1999 1996 2922 1057 1012 1055 1012 3655 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 0 (id = 0)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: test-4
INFO:tensorflow:tokens: [CLS] revenue rose 3 . 9 percent , to $ 1 . 63 billion from $ 1 . 57 billion . [SEP] the mclean , virginia - based company said newspaper revenue increased 5 percent to $ 1 . 46 billion . [SEP]
INFO:tensorflow:input_ids: 101 6599 3123 1017 1012 1023 3867 1010 2000 1002 1015 1012 6191 4551 2013 1002 1015 1012 5401 4551 1012 102 1996 17602 1010 3448 1011 2241 2194 2056 3780 6599 3445 1019 3867 2000 1002 1015 1012 4805 4551 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 0 (id = 0)
```


## Step 5: Freeze Graph

After training and inference has been completed, run ```python model_zoo/bert/freezeTrainedBert.py```
If you have changed paths in any of the previous steps, you will need to change them here also.


## Step 6: Import Graph

The test for importing this graph can be found here:
https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/imports/TFGraphs/BERTGraphTest.java
