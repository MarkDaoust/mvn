#! /bin/bash
echo '#starting'

mkfifo pipe

./test.py $1 &> pipe & < pipe tee testResults.txt

rm pipe

echo '#done'
sleep 1
