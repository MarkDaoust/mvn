
import os
import sys

# external
from pylint import lint
from pylint.reporters.html import HTMLReporter


def main():    
    [dirname,filename] = os.path.split(__file__) 
    
    lintRcPath = os.path.join(dirname,'.lintrc')

    lintFilePath = os.path.join(dirname,'lint.html')
    lintFile = open(lintFilePath,'w')
    
    lint.Run(
        [('--rcfile=%s' % lintRcPath),'mvn'],
        reporter = HTMLReporter(lintFile),
        exit = False,
    )


if __name__ == '__main__':
    main()
