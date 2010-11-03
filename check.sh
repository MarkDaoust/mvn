#! /bin/bash
echo '#starting'

#make a pipe
mkfifo pipe

#check mvar.py, send the output to the pipe,
#and run in the background 
pychecker -#1000 mvar.py &> pipe & 
#on the recieving end of the pipe 
#split the recieved text, printing one copy tothe stdIO, 
#and writing the other copy to a file 
tee checkResults.py < pipe 

rm pipe
echo '#done'
sleep 1
