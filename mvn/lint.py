
import os
import sys

# external
from pylint import lint
from pylint.reporters.html import HTMLReporter


def main(*args):
    args = list(args)
    
    [dirname,filename] = os.path.split(__file__) 
    
    lintRcPath = os.path.join(dirname,'.lintrc')

    lintFilePath = os.path.join(dirname,'lint.html')
    lintFile = open(lintFilePath,'w')
    
    lint.Run(
        args+[('--rcfile=%s' % lintRcPath),'mvn'],
        reporter = HTMLReporter(lintFile),
        exit = False,
    )


if __name__ == '__main__':
    main(*sys.argv[1:])
