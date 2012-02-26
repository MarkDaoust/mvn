#!/usr/bin/env python
'''
**********
Plot Tools
**********
'''

from mvn.matrix import Matrix

import numpy

import pylab
import matplotlib
import  mpl_toolkits.axisartist
from  mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

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


def triax():
    """
    return 3 :py:class:`matplotlib.axes.AxesSubplot`, setup for easy display
    of joint and marginal distributions
    """

    #use a grid of 4x4 cells
    grid = GridSpec(4,4)
    ax = numpy.empty([2,2],dtype = object)

    #create and return the axes
    ax[1,0]   = pylab.subplot(grid[1:,:-1])
    ax[1,0].grid('on')
    
    
    #connect the long axes of the marginals to the mainax
    ax[0,0] = pylab.subplot(grid[0 ,:-1],sharex=ax[1,0])
    ax[0,0].grid('on')
    #reduce the number of ticks on the short axis
    ax[0,0].yaxis.get_major_locator()._nbins = 3
    #and hide the ticks on the long axis    
    pylab.setp(ax[0,0].xaxis.get_ticklabels(),visible=False)    
    
    ax[1,1]  = pylab.subplot(grid[1:, -1],sharey=ax[1,0])
    ax[1,1].grid('on')
    #reduce the number of ticks on the short axis
    ax[1,1].xaxis.get_major_locator()._nbins = 3
    #and hide the ticks on the long axis    
    pylab.setp(ax[1,1].yaxis.get_ticklabels(),visible=False)

    

    return ax
