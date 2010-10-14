#! /bin/bash
echo '#starting'

#make a pipe
mkfifo pipe

#runtest.py with all the input arguments
#send the output and error to the pipe,
#and run in the background 
./test.py $@ &> pipe & 
#on the recieving end of the pipe 
#split the recieved text, printing one copy tothe stdIO, 
#and writing the other copy to a file 
tee testResults.py < pipe 

rm pipe
echo '#done'
sleep 1
