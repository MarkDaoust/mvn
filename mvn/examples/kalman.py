#! /usr/bin/env python
print 'starting'

import os
import sys
import numpy

import matplotlib
#matplotlib.use('cairo')
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

import pylab	

from mvn import Mvn
from mvn.matrix import Matrix

import mvn.plotTools

from collections import OrderedDict

colors = OrderedDict([
    ['Actual'       ,[1,1,0]],
    ['Updated'      ,[0,0,1]],
    ['Noise'        ,[1,0,0]],
    ['Updated+Noise',[1,0,1]],
    ['Measurment'   ,[0,1,0]],
    ['Filter Result',[0,1,1]],
])
    

actualParams={
    'marker':'*',
    'markersize':20,
    'color':colors['Actual'],
}

otherParams={
    'minalpha':0.3
}

class Publisher(object):    
    def __init__(self,targetDir,formats=('png','svg')):
        self.n=0
        self.formats=formats
        self.targetDir=targetDir
        
        try:
            os.stat(self.targetDir)
        except OSError:
            os.mkdir(self.targetDir)


    def publish(self,fig):    
        for format in self.formats:
            fig.savefig("%s/%0.3d.%s" % (self.targetDir,self.n,format),format=format)
        self.n+=1

def seed(path):
    if len(sys.argv)>1:
        seed=int(sys.argv[1])
    else:
        seed=numpy.random.randint(10000)
        print 'seed: %d' % seed
    
    numpy.random.seed(seed)

    open('%s/seed' % path,'w').write(str(seed))

def drawLegend(ax):
    patch=lambda color:matplotlib.patches.Ellipse([0,0],width=0,height=0,facecolor = color)
    
    patches = [patch(color) for [name,color] in colors.iteritems()]

    ax.legend(
        patches,list(colors.keys()),
        loc='lower right'
    )
    
def newAx(fig,transform = Matrix.eye(2)):
    fig.clear()

    axgrid = GridSpec(1,1)
    
    #get axes
    ax = pylab.subplot(
        axgrid[:,:],
        projection = 'custom', 
        transform = transform,
    )
    
    ax.autoscale(False)

#    ax.set_xticks(numpy.arange(-10.,35.,5.))
#    ax.set_yticks(numpy.arange(-10.,35.,5.))
    
    ax.set_xlim([-5,20])
    ax.set_ylim([-5,10])    
    
    ax.xaxis.set_major_locator(MultipleLocator(5))

    ax.grid('on')
    drawLegend(ax)    
    
    return ax

if __name__=='__main__':


    ## figure setup
    
    #directory for resulting figures
    path = 'kalman'    
    #seed the rng so results are reproducible.
    seed(path)
    #create publisher
    P = Publisher(path)
    #create figure
    fig = pylab.figure(figsize = (7,7))


    ## kalman filter parameters

    #the actual, hidden state
    actual=numpy.array([[0,5]])

    #the sensor
    sensor=Mvn(vectors=[[1,0],[0,1]],var=[0.5,numpy.inf])

    #the system noise
    noise=Mvn(vectors=[[1,-0.2],[0.2,1]],var=numpy.array([0.5,1])**2)

    #the shear transform to move the system forward
    transform=Matrix([[1,0],[0.5,1]])

    filtered=sensor.measure(actual)


    ## initial plot

    ax = newAx(fig)
    
    #plot the initial actual position
    ax.plot(actual[:,0],actual[:,1],**actualParams)
    ax.set_title('Kalman Filtering: Start')
    pylab.xlabel('Position')
    pylab.ylabel('Velocity')    
    P.publish(fig)



    #measure the actual position, and plot the measurment
    filtered.plot(facecolor=colors['Measurment'],**otherParams)
    ax.set_title('Initialize to first measurment')
    pylab.xlabel('Position')
    pylab.ylabel('Velocity')
    P.publish(fig)

    for n in range(8):
 
        ## plot immediately after the step foreward

        #create a transformed axis        
        ax = newAx(fig,transform)
        
        #update the system
        actual=actual*transform
        filtered=filtered*transform
    
        #plot the updated system
        ax.plot(actual[:,0],actual[:,1],**actualParams)
        filtered.plot(facecolor=colors['Updated'],**otherParams)
        ax.set_title('Update')        
        pylab.xlabel('Position')
        pylab.ylabel('Velocity')
        P.publish(fig)

        #realign the axes
        ax = newAx(fig)
    
        #re-plot the filter result
        filtered.plot(facecolor=colors['Updated'],**otherParams) 

        #add noise and plot the actual and filtered values
        actual=noise+actual
        filtered=noise+filtered
               
        actual.plot(facecolor=colors['Noise'],**otherParams)
        filtered.plot(facecolor=colors['Noise'],**otherParams)

        # sample the position of the actual distribution, to find it's new position
        actual=actual.sample()
        ax.plot(actual[:,0],actual[:,1],**actualParams)
        
        ax.set_title('Add process noise')    
        pylab.xlabel('Position')
        pylab.ylabel('Velocity')
        P.publish(fig)

        ax = newAx(fig)

        ax.plot(actual[:,0],actual[:,1],**actualParams)
        filtered.plot(facecolor=colors['Updated+Noise'],**otherParams)
        ax.set_title('Add process noise')    
        pylab.xlabel('Position')
        pylab.ylabel('Velocity')
        P.publish(fig)

        measure=sensor.measure(actual)
        measure.plot(facecolor=colors['Measurment'],**otherParams)
        ax.set_title('Measure')
        P.publish(fig)
        
        
        filtered=filtered&measure
        filtered.plot(facecolor=colors['Filter Result'],**otherParams)
        ax.set_title('Merge')    
        pylab.xlabel('Position')
        pylab.ylabel('Velocity')
        P.publish(fig)

    
        ax = newAx(fig)
        
        ax.plot(actual[:,0],actual[:,1],**actualParams)
        filtered.plot(facecolor=colors['Filter Result'],**otherParams) 
        pylab.xlabel('Position')
        pylab.ylabel('Velocity')
        ax.set_title('Merge')    
        P.publish(fig)

#    os.system('convert -limit memory 32 -delay 100 %s/*.png kalman.gif' % path)    
    os.system('convert -delay 150 %s/*.png kalman.gif' % path)

