#!/usr/bin/env python

import pylab

import numpy

from mvn import Mvn

from matplotlib.gridspec import GridSpec


def triax():
    """
    return 3 :py:class:`matplotlib.axes.AxesSubplot`, stacked horizontally
    """
    #create a figures of the apropriate size    
    pylab.figure(1, figsize=(12,4))

    #use a grid of 4x4 cells
    grid = GridSpec(1,3)

    #create axis
    ax0  = pylab.subplot(grid[:,0])
    ax1  = pylab.subplot(grid[:,1])
    ax2  = pylab.subplot(grid[:,2])

    return ax0,ax1,ax2


def main():
    axes = triax()    
    
        
    N=int(10.0**(1+2*numpy.random.rand()))
        
    for dims in range(3):
        ax = axes[dims]
        ax.set_title("Mvn.rand((%s,2))" % dims)
        for n in range(N):
            M = Mvn.rand((dims,2))
            M.plot(ax,facecolor = numpy.random.rand(3))
            
    
    pylab.tight_layout()
    pylab.show()

if __name__ == '__main__':
    main()
