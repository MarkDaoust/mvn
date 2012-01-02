"""
plot for the main page of the documentation

.. plot:: ../plots/introPlot.py main    
"""
import numpy

import matplotlib.pyplot as plt

from mvar import Mvar
from mvar.plotTools import AxGrid

def introPlot():
    #create the figures and axes    
    plt.figure(1, figsize=(7,7))
    
    grid = AxGrid(4,4,spacing = 0.025)
    
    scatterAX= grid[1:,:-1]
    xhistAX  = grid[ 0,:-1]
    yhistAX  = grid[1:, -1]
    
    xhistAX.xaxis.set_ticklabels('')
    yhistAX.yaxis.set_ticklabels('')
    
    #generate some random data
    N = 500
    xy = numpy.dot(numpy.random.randn(N,2),numpy.random.randn(2,2));
    
    x = xy[:,0]
    y = xy[:,1]
    
    #generate mvars from the data
    XY=Mvar.fromData(xy)
        
    X = XY[:,0]
    Y = XY[:,1]
    
    #add the ellipse and scatter plot
    scatterAX.add_artist(XY.patch(alpha = 0.33,nstd=2,color='r',zorder = 0))
    scatterAX.scatter(x,y,zorder = 1);
    
    #turn off the autoscaling so the axes don't move 
    #when I add the projection lines
    scatterAX.autoscale(False)
    
    #number of histogram bins
    nbins=50
    
    #draw the two histograms and gaussians
    xlim = scatterAX.get_xlim()
    xhistAX.set_xlim(xlim)
    xbins = numpy.linspace(xlim[0],xlim[1],nbins)
    xhistAX.hist(x, bins=xbins, orientation='vertical')
    xcoord = numpy.linspace(xlim[0],xlim[1],500)[:,None]
    xhistAX.plot(xcoord,N*X.density(xcoord)*numpy.diff(xlim)/nbins,color='r')
    
    ylim =scatterAX.get_ylim()
    yhistAX.set_ylim(ylim)
    ybins = numpy.linspace(ylim[0],ylim[1],nbins)
    yhistAX.hist(y, bins=ybins, orientation='horizontal')
    ycoord = numpy.linspace(ylim[0],ylim[1],500)[:,None]
    yhistAX.plot(N*Y.density(ycoord)*numpy.diff(ylim)/nbins,ycoord,color='r')
    
    #draw the limit lines
    scatterAX.plot(numpy.concatenate([X.mean+2*X.width(),X.mean+2*X.width()]),ylim,color='r')
    scatterAX.plot(numpy.concatenate([X.mean-2*X.width(),X.mean-2*X.width()]),ylim,color='r')
    scatterAX.plot(xlim,numpy.concatenate([Y.mean+2*Y.width(),Y.mean+2*Y.width()]),color='r')
    scatterAX.plot(xlim,numpy.concatenate([Y.mean-2*Y.width(),Y.mean-2*Y.width()]),color='r')
    
    xhistAX.plot(numpy.concatenate([X.mean+2*X.width(),X.mean+2*X.width()]),xhistAX.get_ylim(),color='r')
    xhistAX.plot(numpy.concatenate([X.mean-2*X.width(),X.mean-2*X.width()]),xhistAX.get_ylim(),color='r')
    
    yhistAX.plot(yhistAX.get_xlim(),numpy.concatenate([Y.mean+2*Y.width(),Y.mean+2*Y.width()]),color='r')
    yhistAX.plot(yhistAX.get_xlim(),numpy.concatenate([Y.mean-2*Y.width(),Y.mean-2*Y.width()]),color='r')
    
    #show the plot
    plt.show()
