#! /usr/bin/env python
"""
.. plot:: ./examples/marginals.py main
"""
import numpy
import pylab

from matplotlib.gridspec import GridSpec

from mvar import Mvar

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

def hist(red,blue,ax,orientation):
    """
    compare theoretical density to histogram
    """
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
        ax=ax, 
        orientation=orientation,
        zorder = 1,
        **plotParams
    )
    
    #reduce the number of ticks on the short axis
    shortax.get_major_locator()._nbins = 3
    #and hide the ticks on the long axis    
    pylab.setp(longax.get_ticklabels(),visible=False)

def triax():
    """
    return 3 :py:class:`matplotlib.axes.AxesSubplot`, setup for easy display
    of joint and marginal distributions
    """
    #create a figures of the apropriate size    
    pylab.figure(1, figsize=(6,6))

    #use a grid of 4x4 cells
    grid = GridSpec(4,4)

    #create and return the axes
    mainax      = pylab.subplot(grid[1:,:-1])
    #connect the long axes of the marginals to the mainax
    topax   = pylab.subplot(grid[0 ,:-1],sharex=mainax)
    sideax  = pylab.subplot(grid[1:, -1],sharey=mainax)

    return mainax,topax,sideax


def main():
    """
    demonstrate marginal distributions
    """
    N=numpy.round(10.0**(1+3.5*numpy.random.rand()))
    
    #generate a random mvar
    red=Mvar.rand(ndims=2)
    
    #sample some data
    blue = red.sample(N)

    #create axes for plotting
    mainax,topax,sideax = triax()
        
    #do the main plot
    mainax.scatter(blue[:,0],blue[:,1],zorder = 0,alpha = alpha);
    red.plot(ax = mainax, nstd=2, zorder = 1,label=r'$\pm 2 \sigma$',**plotParams)    
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
    mainax.set_xlabel('>>> red',font)
    topax.set_title('>>> red[:,0]',font)
    sideax.set_title('>>> red[:,1]',font)
    
    #draw the figure
    pylab.show()
    
if __name__ == '__main__':
    main()