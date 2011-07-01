#! /bin/bash

tee<pipe testresult.txt &
./runTests.py &> pipe 

git diff testresult.txt | tee testresult.diff
