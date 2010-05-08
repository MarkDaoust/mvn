#! /bin/bash
echo starting
mkfifo pipe
./run_test.py &> pipe &
tee test_results.py < pipe
rm pipe
echo done
sleep 1
