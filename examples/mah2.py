#!/usr/bin/env python

"""
.. plot:: ./examples/mah.py main
"""
import itertools

import pylab
from pylab import subplot

import numpy

import matplotlib
from matplotlib.gridspec import GridSpec

import  mpl_toolkits.axisartist
from  mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

from mvn import Mvn
from mvn.matrix import Matrix
from mvn.decorate import curry


class my_axis(mpl_toolkits.axisartist.Axes):
    def __init__(self,*args,**kwargs):
        try:
            transform = kwargs.pop('transform')
        except KeyError:
            transform = Matrix.eye(2)
            
        if isinstance(transform,GridHelperCurveLinear):
            assert 'Itransform' not in kwargs,(
                'no Itransform when transform is a %s' % 
                type(transform)            
            )
            grid_helper = transform
        elif callable(transform):
            grid_helper =  GridHelperCurveLinear([
                transform,
                kwargs.pop('Itransform')
            ])
        else:
            transform = Matrix(transform)

            try:
                Itransform = kwargs.pop('Itransform')
            except KeyError:
                Itransform = transform**(-1)
            
            grid_helper =  GridHelperCurveLinear([
                self.makeTransform(transform),
                self.makeTransform(Itransform)
            ])
            
        kwargs['grid_helper']=grid_helper
        mpl_toolkits.axisartist.Axes.__init__(self,*args,**kwargs)
        
    @curry
    def makeTransform(self,M, x, y):
        x = Matrix(x)
        y = Matrix(y)
        xy = numpy.hstack([x.T,y.T])
        xy = xy*M
        return xy[:,0].squeeze(),xy[:,1].squeeze()


###########################
# DANGER!: Monkey Patching#

matplotlib.projections.projection_registry._all_projection_types[
        'custom'
    ] = my_axis
    
# End Monkey Patch        #
###########################

            


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
    print alt
    alt = numpy.array(alt)
    return alt    


def main():
    N=75#numpy.round(10.0**(1.5+1*numpy.random.rand()))

    #generate data
    red = Mvn.rand(2)
    blue = red.sample(N)
    red = Mvn.fromData(blue)

    #create figure
    fig = pylab.figure(1, figsize=(7, 7))
    fig.suptitle('Mahalabois Distance')
    
    axgrid = GridSpec(2,2)
    ax = numpy.empty([2,2],dtype = object)

    #get axes
    ax[0,0] = subplot(axgrid[0,0])
    ax[0,0].axis('equal')
    ax[0,0].grid('on')
    ax[0,0].set_title('red')
    
    


                    
    @curry
    def forewardTransform(center,M, x, y):

        center = center.squeeze()

        x = Matrix(x)
        y = Matrix(y)
        xy = numpy.hstack([x.T,y.T])
        xy = xy - center[None,:]

        xy = numpy.array(xy*M)

        mags = (numpy.array(xy)**2).sum(1)[:,None]**0.5
        dirs = xy/mags

        xy = dirs*mags**2
        
        return xy[:,0].squeeze(),xy[:,1].squeeze()

    @curry
    def reverseTransform(center,M, x, y):

        center = center.squeeze()

        x = Matrix(x)
        y = Matrix(y)
        xy = numpy.hstack([x.T,y.T])
        xy = xy - center[None,:]

        xy = numpy.array(xy*M)

        mags = (numpy.array(xy)**2).sum(1)[:,None]**0.5
        dirs = xy/mags

        xy = dirs*mags**0.5 + center
        
        return xy[:,0].squeeze(),xy[:,1].squeeze()


    foreward = forewardTransform(red.mean,red.transform(-1))
    reverse = reverseTransform(red.mean,red.transform(1))
    
    grid_helper =  GridHelperCurveLinear([
        foreward,
        reverse
    ])
    
    
    
    

    ax[1,0] = subplot(axgrid[1,0],
        projection = 'custom', 
        transform = grid_helper
    )    
    ax[1,0].axis('equal')
    ax[1,0].grid('on')
    ax[1,0].set_title('red/red')
    
    ax[0,1] = subplot(axgrid[0,1])
    ax[0,1].grid('on')
    ax[0,1].set_title('red.mah2().cdf()')
    ax[0,1].set_ylim([0,1])
    
    ax[1,1] = pylab.subplot(axgrid[1,1],sharex=ax[0,1])#,sharey=ax[0,1])
    ax[1,1].grid('on')
    ax[1,1].set_title('red.mah2().pdf()')
    
    scatter(ax[0,0],red,blue)

  #  blue0 = Matrix(blue)/red
  #  blue0 = blue0.array()
    red0 = red/red
    
    blue0 = foreward(blue[:,0],blue[:,1])
    blue0 = numpy.hstack([blue0[0][:,None],blue0[1][:,None]])
    
    scatter(ax[1,0],red0,blue0)

    mah2 = red.mah2(blue)
    mah2.sort()
    ax[0,1].hlines(numpy.arange(mah2.size,dtype = float)/mah2.size,0,mah2,color = 'b',alpha = alpha)
    ax[0,1].scatter(mah2,numpy.arange(mah2.size,dtype = float)/mah2.size,color = 'b',alpha = alpha)
    x = numpy.linspace(*ax[0,1].get_xlim(),num=500)    
    ax[0,1].plot(x,red.mah2().cdf(x),color = 'r',linewidth = 2,alpha = alpha)
    hist(red.mah2().pdf,red.mah2(blue),ax[1,1])

    ax[0,1].set_xlim([0,None])
    ax[1,1].set_ylim([0,None])
    
    pylab.show()

if __name__ == '__main__':
    main()
