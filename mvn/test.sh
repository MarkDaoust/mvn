#! /bin/bash

mv testresult.txt testresult_old.txt

./runTests.py $@ 2>&1 | tee testresult.txt &

diff testresult_old.txt testresult.txt > testresult.diff
