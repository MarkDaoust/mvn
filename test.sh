#! /bin/bash

tee<pipe testresult.txt &
./runTests.py $@ &> pipe 
echo '*************************************************************************'
git diff testresult.txt | tee testresult.diff
