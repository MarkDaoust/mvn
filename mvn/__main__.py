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
    
    lintRcPath = os.path.join(dirname,'.lintrc')

    lintFilePath = os.path.join(dirname,'lint.html')
    lintFile = open(lintFilePath,'w')
    
    lint.Run(
        ['--rcfile=%s' % lintRcPath,'mvn'],
        reporter = HTMLReporter(lintFile)
    )

    ## run nosetests
    test.main([])

    ## run sphinx
    



main()