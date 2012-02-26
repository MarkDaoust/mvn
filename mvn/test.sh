#! /bin/bash

mv testresult.txt testresult_old.txt

tee<pipe testresult.txt &
./runTests.py $@ &> pipe 

diff testresult_old.txt testresult.txt > testresult.diff
