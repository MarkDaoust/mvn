import os
import sys
import subprocess

#import nose

def main(argv = None):

    if argv is None:
        argv = []

    [testPath,filename] = os.path.split(__file__) 
    [mvnPath,_] = os.path.split(testPath)

    resultPath = os.path.join(testPath,'results.txt')    
    
    targets = [mvnPath,os.path.join(testPath+'/unit.py')]
    args = ['--with-coverage','--cover-package=mvn','--with-doctest']

    
    #    nose.run(argv = args+targets)

    tee = subprocess.Popen(
        ['tee',resultPath],
         stdin = subprocess.PIPE,
         stdout = sys.stdout
    )
    
    tests = subprocess.Popen(
        ['nosetests']+args+targets,
        stdout = tee.stdin,
        stderr = tee.stdin
    )

    tests.communicate()
