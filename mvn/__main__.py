# builtin
import os
import sys


# local
import mvn.test
import mvn.lint
import mvn.sphinx

def main():              
    ## run pylint
    mvn.lint.main()

    ## run nosetests
    mvn.test.main()

    ## run sphinx
    mxn.sphinx.main()


main()












