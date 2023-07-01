#!/bin/sh
i=1
for f1 in ./dylib/*
do
	for f2 in ./TestCases/*
	do
		v1=${f1:0-16:10}
		v2=${f2:0-8:2}
		echo compete $f1 $f2 ./compete_result/${v1}_${v2}.txt
		./Compete $f1 $f2 "./compete_result/${v1}_${v2}.txt" 1
	done
done
