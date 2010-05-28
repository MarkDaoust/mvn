#! /bin/bash
echo '#starting'
mkfifo pipe
./comment_doctest.py ./mvar.py $1 &> pipe &
< pipe tee test_results.py
rm pipe
echo '#done'
sleep 1
