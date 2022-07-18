#!/usr/bin/env bash

set -e
set -u

echo "dataset,depth,leaf_count,internal_count,class_count,duration,batch_size"
for sz in 1 6
do
	for data_set in heart breast steel spam2 spam
	do
	    res=$(./target/release/homdte --dir=data/"$data_set" --input-size="$sz" --parallel)
	    echo "$res,$sz"
	done

	for data_set in fake_hou fake_art
	do
        res=$(./target/release/homdte --dir=data/"$data_set" --input-size="$sz" --parallel --artificial)
	    echo "$res,$sz"
	done
done
