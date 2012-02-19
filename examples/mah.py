#!/usr/bin/env python

"""
.. plot:: ./examples/mah.py main
"""
import itertools


import numpy

import pylab

from matplotlib.gridspec import GridSpec

from mvar import Mvar
from mvar.matrix import Matrix

import mvar.plotTools


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
    x = numpy.linspace(0,blue.max()+10,500)
    ax.axis('tight')
    ax.set_autoscale_on(False)
    ax.fill_between(x,
        red(x),0,        
        zorder = 1,
        **plotParams
    )

def scatter(ax,red,blue):
    blue = blue.array()
    
    stemmed = stem(blue,center=red.mean)
    ax.plot(stemmed[:,0],stemmed[:,1],zorder = 0,alpha = alpha)
    
    red.plot(ax,**plotParams)
    ax.scatter(blue[:,0],blue[:,1],alpha = alpha)
#    ax.plot([0,red.mean[0,0]],[0,red.mean[0,1]],color='k',zorder = 2,linewidth=2)  
    ax.scatter(red.mean[:,0],red.mean[:,1],color='y',marker='*',zorder = 3,s=300,edgecolor='k')
    
def alternate(iterables):
    iterables = [iter(I) for I in iterables]
    while True:
        for I in iter(iterables):
            yield I.next()
    
def stem(data,center):
    # flatten the center
    center = numpy.ravel(center)
    centers = itertools.repeat(center)
    alt = list(alternate([data,centers]))    
    alt = numpy.array(alt)
    return alt    


def cumulative(ax,red,blue):
    mah = red.mah(blue)
    mah.sort()

    fraction = numpy.arange(mah.size,dtype = float)/mah.size

    ax.hlines(fraction,0,mah,color = 'b',alpha = alpha)
    ax.scatter(mah,fraction,color = 'b',alpha = alpha)
    
    x = numpy.linspace(*ax.get_xlim(),num=500)    
    ax.plot(x,red.mah().cdf(x),color = 'r',linewidth = 2,alpha = alpha)    

def setupAxes(transform):
    #create axes grid
    axgrid = GridSpec(2,2)
    ax = numpy.empty([2,2],dtype = object)

    #get axes
    ax[0,0] = pylab.subplot(axgrid[0,0])
    ax[0,0].axis('equal')
    ax[0,0].grid('on')
    ax[0,0].set_title('red')
    
    ax[1,0] = pylab.subplot(axgrid[1,0],
        projection = 'custom', 
        transform = transform
    )    
    ax[1,0].axis('equal')
    ax[1,0].grid('on')
    ax[1,0].set_title('red/red')
    
    ax[0,1] = pylab.subplot(axgrid[0,1])
    ax[0,1].grid('on')
    ax[0,1].set_title('red.mah().cdf()')
    ax[0,1].set_ylim([0,1])
    
    ax[1,1] = pylab.subplot(axgrid[1,1],sharex=ax[0,1])
    ax[1,1].grid('on')
    ax[1,1].set_title('red.mah().pdf()')
    
    return ax

def main():

    #generate data
    N=75
    red = Mvar.rand(2)
    blue = Matrix(red.sample(N))
    red = Mvar.fromData(blue)

    #create figure  
    fig = pylab.figure(1, figsize=(7, 7))
    fig.suptitle('Mahalabois Distance')

    ax = setupAxes(red.transform(-1))

    # scatter plot of the origional data        
    scatter(ax[0,0],red,blue)

    # scatter plot of the normalized data
    scatter(ax[1,0],red/red,blue/red)

    # draw the cumulative distribution
    cumulative(ax[0,1],red,blue)

    # draw the histogram
    hist(red.mah().pdf,red.mah(blue),ax[1,1])

    ax[0,1].set_xlim([0,None])
    ax[1,1].set_ylim([0,None])
    
    pylab.show()

if __name__ == '__main__':
    main()