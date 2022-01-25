data_folder=$1 # data folder

echo "****** RQ2 - Variation 1 ******" # token + sentence
echo "*** 100 ***"
python train.py $data_folder/rq2/en_train_100.json 41 true false
python train.py $data_folder/rq2/en_train_100.json 42 true false
python train.py $data_folder/rq2/en_train_100.json 43 true false

echo "*** 90 ***"
python train.py $data_folder/rq2/en_train_90.json 41 true false
python train.py $data_folder/rq2/en_train_90.json 42 true false
python train.py $data_folder/rq2/en_train_90.json 43 true false

echo "*** 80 ***"
python train.py $data_folder/rq2/en_train_80.json 41 true false
python train.py $data_folder/rq2/en_train_80.json 42 true false
python train.py $data_folder/rq2/en_train_80.json 43 true false

echo "*** 70 ***"
python train.py $data_folder/rq2/en_train_70.json 41 true false
python train.py $data_folder/rq2/en_train_70.json 42 true false
python train.py $data_folder/rq2/en_train_70.json 43 true false

echo "*** 60 ***"
python train.py $data_folder/rq2/en_train_60.json 41 true false
python train.py $data_folder/rq2/en_train_60.json 42 true false
python train.py $data_folder/rq2/en_train_60.json 43 true false

echo "*** 50 ***"
python train.py $data_folder/rq2/en_train_50.json 41 true false
python train.py $data_folder/rq2/en_train_50.json 42 true false
python train.py $data_folder/rq2/en_train_50.json 43 true false

echo "*** 40 ***"
python train.py $data_folder/rq2/en_train_40.json 41 true false
python train.py $data_folder/rq2/en_train_40.json 42 true false
python train.py $data_folder/rq2/en_train_40.json 43 true false

echo "*** 30 ***"
python train.py $data_folder/rq2/en_train_30.json 41 true false
python train.py $data_folder/rq2/en_train_30.json 42 true false
python train.py $data_folder/rq2/en_train_30.json 43 true false

echo "*** 20 ***"
python train.py $data_folder/rq2/en_train_20.json 41 true false
python train.py $data_folder/rq2/en_train_20.json 42 true false
python train.py $data_folder/rq2/en_train_20.json 43 true false

echo "*** 10 ***"
python train.py $data_folder/rq2/en_train_10.json 41 true false
python train.py $data_folder/rq2/en_train_10.json 42 true false
python train.py $data_folder/rq2/en_train_10.json 43 true false


echo "****** RQ2 - Variation 2 ******" # token + document
echo "*** 100 ***"
python train.py $data_folder/rq2/en_train_100.json 41 false true
python train.py $data_folder/rq2/en_train_100.json 42 false true
python train.py $data_folder/rq2/en_train_100.json 43 false true

echo "*** 90 ***"
python train.py $data_folder/rq2/en_train_90.json 41 false true
python train.py $data_folder/rq2/en_train_90.json 42 false true
python train.py $data_folder/rq2/en_train_90.json 43 false true

echo "*** 80 ***"
python train.py $data_folder/rq2/en_train_80.json 41 false true
python train.py $data_folder/rq2/en_train_80.json 42 false true
python train.py $data_folder/rq2/en_train_80.json 43 false true

echo "*** 70 ***"
python train.py $data_folder/rq2/en_train_70.json 41 false true
python train.py $data_folder/rq2/en_train_70.json 42 false true
python train.py $data_folder/rq2/en_train_70.json 43 false true

echo "*** 60 ***"
python train.py $data_folder/rq2/en_train_60.json 41 false true
python train.py $data_folder/rq2/en_train_60.json 42 false true
python train.py $data_folder/rq2/en_train_60.json 43 false true

echo "*** 50 ***"
python train.py $data_folder/rq2/en_train_50.json 41 false true
python train.py $data_folder/rq2/en_train_50.json 42 false true
python train.py $data_folder/rq2/en_train_50.json 43 false true

echo "*** 40 ***"
python train.py $data_folder/rq2/en_train_40.json 41 false true
python train.py $data_folder/rq2/en_train_40.json 42 false true
python train.py $data_folder/rq2/en_train_40.json 43 false true

echo "*** 30 ***"
python train.py $data_folder/rq2/en_train_30.json 41 false true
python train.py $data_folder/rq2/en_train_30.json 42 false true
python train.py $data_folder/rq2/en_train_30.json 43 false true

echo "*** 20 ***"
python train.py $data_folder/rq2/en_train_20.json 41 false true
python train.py $data_folder/rq2/en_train_20.json 42 false true
python train.py $data_folder/rq2/en_train_20.json 43 false true

echo "*** 10 ***"
python train.py $data_folder/rq2/en_train_10.json 41 false true
python train.py $data_folder/rq2/en_train_10.json 42 false true
python train.py $data_folder/rq2/en_train_10.json 43 false true


echo "****** RQ2 - Variation 3 ******" # token + sentence + document
echo "*** 100 ***"
python train.py $data_folder/rq2/en_train_100.json 41 true true
python train.py $data_folder/rq2/en_train_100.json 42 true true
python train.py $data_folder/rq2/en_train_100.json 43 true true

echo "*** 90 ***"
python train.py $data_folder/rq2/en_train_90.json 41 true true
python train.py $data_folder/rq2/en_train_90.json 42 true true
python train.py $data_folder/rq2/en_train_90.json 43 true true

echo "*** 80 ***"
python train.py $data_folder/rq2/en_train_80.json 41 true true
python train.py $data_folder/rq2/en_train_80.json 42 true true
python train.py $data_folder/rq2/en_train_80.json 43 true true

echo "*** 70 ***"
python train.py $data_folder/rq2/en_train_70.json 41 true true
python train.py $data_folder/rq2/en_train_70.json 42 true true
python train.py $data_folder/rq2/en_train_70.json 43 true true

echo "*** 60 ***"
python train.py $data_folder/rq2/en_train_60.json 41 true true
python train.py $data_folder/rq2/en_train_60.json 42 true true
python train.py $data_folder/rq2/en_train_60.json 43 true true

echo "*** 50 ***"
python train.py $data_folder/rq2/en_train_50.json 41 true true
python train.py $data_folder/rq2/en_train_50.json 42 true true
python train.py $data_folder/rq2/en_train_50.json 43 true true

echo "*** 40 ***"
python train.py $data_folder/rq2/en_train_40.json 41 true true
python train.py $data_folder/rq2/en_train_40.json 42 true true
python train.py $data_folder/rq2/en_train_40.json 43 true true

echo "*** 30 ***"
python train.py $data_folder/rq2/en_train_30.json 41 true true
python train.py $data_folder/rq2/en_train_30.json 42 true true
python train.py $data_folder/rq2/en_train_30.json 43 true true

echo "*** 20 ***"
python train.py $data_folder/rq2/en_train_20.json 41 true true
python train.py $data_folder/rq2/en_train_20.json 42 true true
python train.py $data_folder/rq2/en_train_20.json 43 true true

echo "*** 10 ***"
python train.py $data_folder/rq2/en_train_10.json 41 true true
python train.py $data_folder/rq2/en_train_10.json 42 true true
python train.py $data_folder/rq2/en_train_10.json 43 true true
