#! /usr/bin/env python
"""
.. plot:: ./examples/blend.py main
"""

import pylab

from mvar import Mvar

def main():
    """
    demonstrate the `covariance intersection algorithm
    <http://en.wikipedia.org/wiki/Kalman_filter#Update>`_ 
    """
    pylab.figure(1, figsize=(5,5))
    
    red  = Mvar.rand(shape=2)    
    blue = Mvar.rand(shape=2)
    
    magenta = red & blue
    
    red.plot(    facecolor = 'r',edgecolor = 'k',slope=1)
    blue.plot(   facecolor = 'b',edgecolor = 'k',slope=1)
    magenta.plot(facecolor = 'm',edgecolor = 'k',slope=1)
    
    
    pylab.xlabel('Magenta = Red & Blue')    
    pylab.show()

if __name__ == '__main__':
    main()