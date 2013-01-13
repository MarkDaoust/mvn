#! /usr/bin/env python
"""
.. plot:: ./examples/marginals.py main
"""
import numpy
import pylab


from mvn import Mvn
from mvn.plotTools import triax

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
    'edgecolor':'k',
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
        
    #find the 'long' and 'short' axes
    xaxis = ax.xaxis
    yaxis = ax.yaxis
    
    if orientation == 'horizontal':
        longax,shortax = yaxis,xaxis
    elif orientation == 'vertical':
        longax,shortax = xaxis,yaxis

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
        axis = ax, 
        orientation = orientation,
        zorder = 1,
        **plotParams
    )


def main():
    """
    demonstrate marginal distributions
    """
    N=numpy.round(10.0**(1+2.5*numpy.random.rand()))
    
    #generate a random mvn
    red=Mvn.rand(2)
    
    #sample some data
    blue = red.sample(N)

    #create a figures of the apropriate size    
    pylab.figure(1, figsize=(6,6))

    #create axes for plotting
    axes = triax()
    mainax = axes[1,0]
    topax  = axes[0,0]
    sideax = axes[1,1]
        
    #do the main plot
    mainax.scatter(blue[:,0],blue[:,1],zorder = 0,alpha = alpha);
    red.plot(axis = mainax, nstd=2, zorder = 1,label=r'$\pm 2 \sigma$',**plotParams)    
    mainax.legend()    
    
    #freeze the main axes
    mainax.autoscale(False)
    topax.set_autoscalex_on(False)
    sideax.set_autoscaley_on(False)
    
    #plot the histograms
    hist(red[:,0],blue[:,0],topax,orientation = 'vertical')
    hist(red[:,1],blue[:,1],sideax,orientation = 'horizontal')
    
    #set the main title
    font = {'size':12}
    mainax.set_xlabel('red',font)
    topax.set_title('red[:,0]',font)
    sideax.set_title('red[:,1]',font)
    
    #draw the figure
    pylab.show()
    
if __name__ == '__main__':
    main()
