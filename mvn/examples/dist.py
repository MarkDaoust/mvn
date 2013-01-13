#!/usr/bin/env python

"""
.. plot:: ./examples/dist.py main
"""

import pylab

from mvn import Mvn


from matplotlib.gridspec import GridSpec

alpha = 0.5
"""
standard alpha for all plot elements
"""

nbins=50
"""
number of bins for the histograms
"""

plotParams = {
    'alpha':alpha,
    'facecolor':'r',
}
"""
plot params used in this example
"""


def hist(red,blue,ax=None,orientation='vertical'):
    """
    compare theoretical density to histogram
    """
    if ax is None:
        ax=pylab.gca()
        
    #draw the histogram
    ax.hist(
        blue, 
        bins = nbins,
        orientation=orientation,
        zorder = 0,
        alpha = alpha,
        normed=True, # convert from bin conts to PDF
        histtype='stepfilled'
    )
    
    #plot the normal distribution
    red.plot(
        axis=ax, 
        orientation=orientation,
        zorder = 1,
        **plotParams
    )
    

def main():
    
    #get axis
    ax0 = pylab.subplot(1,2,1)
    ax1 = pylab.subplot(1,2,2)    
    
    #get data
    A = Mvn.rand(2)
    data = A.sample(10000)
    
    #squish the data to spherical variance    
    deltas = ((data-A.mean)/A).array()
    
    #calculate the errors in the squished space
    errors = (deltas**2).sum(1)
    
    print errors
    
    hist(A.dist2(),errors,ax0)
    
    
    pylab.show()

if __name__ == '__main__':
    main()
