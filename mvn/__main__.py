# builtin
import os
import sys
import subprocess

# external
from pylint import lint
from pylint.reporters.html import HTMLReporter

#import sphinx

# local
import mvn.test


def main():
    ## run pylint
    [dirname,filename] = os.path.split(__file__) 
    
    lintFileName = os.path.join(dirname,'lint.html')
    lintFile = open(lintFileName,'w')
    
    lint.Run(
        ['--include-ids=y','--disable=I0011','mvn'],
        reporter = HTMLReporter(lintFile)
    )

    ## run nosetests
    test.main([])

    ## run sphinx
    



main()