#! /usr/bin/env python
print 'starting'


from pylab import *
from mvar import Mvar,Matrix

import numpy

#numpy.random.seed(12)


class publisher(object):    
    def __init__(self):
        self.n=0

    def publish(self):    
        xticks(range(-20,20,5))
        yticks(range(-20,20,5))
        grid(True)

        xlim(-2.5,17.5)
        xlabel('Position')
        ylim(-2.5,12.5)
        ylabel('Velocity')

        E=lambda **kwargs:matplotlib.patches.Ellipse([0,0],width=0,height=0,**kwargs)


        bottom_left=3
        legend(
            [
                E(fc=[1,0,1]),
                E(fc=[0,1,0]),
                E(fc=[1,0,0]),
                E(fc=[1,1,0]),
                E(fc=[0,0,1]),
                E(fc=[0,1,1]),
            ],[
                'Actual',
                'Updated',
                'Noise',
                'Updated + Noise',
                'Measurment',
                'Filter Result'                
            ],
            loc='upper left'
        )

        savefig("art/%0.3d" % self.n,format='png')
        self.n+=1

def do(real,filtered,sensor,noise,transform):
    #update the system
    real=real*transform
    filtered=filtered*transform

    plot(real[:,0],real[:,1],'v',color=[1,0,1],ms=20)
    ax.add_artist(filtered.patch(minalpha=0.1,slope=0.5,facecolor=[0,1,0]))
    title('Update')
    P.publish()

    ax.clear()

    ax.add_artist(filtered.patch(minalpha=0.1,slope=0.5,facecolor=[0,1,0])) 

    real=noise+real
    filtered=noise+filtered

    ax.add_artist(real.patch(minalpha=0.1,slope=0.5,facecolor=[1,0,0]))
    ax.add_artist(filtered.patch(minalpha=0.1,slope=0.5,facecolor=[1,0,0]))
    
    real=real.sample()

    plot(real[:,0],real[:,1],'v',color=[1,0,1],ms=20)
    
    title('Add process noise')    
    P.publish()

    ax.clear()

    plot(real[:,0],real[:,1],'v',color=[1,0,1],ms=20)
    ax.add_artist(filtered.patch(minalpha=0.1,slope=0.5,facecolor=[1,1,0]))
    title('Add process noise')    
    P.publish()

    measure=sensor.measure(real)
    ax.add_artist(measure.patch(minalpha=0.1,slope=0.5,facecolor=[0,0,1]))
    title('Measure')
    P.publish()

    filtered=filtered&measure
    ax.add_artist(filtered.patch(minalpha=0.1,slope=0.5,facecolor=[0,1,1]))
    title('Merge')    
    P.publish()

    ax.clear()
    plot(real[:,0],real[:,1],'v',color=[1,0,1],ms=20)
    ax.add_artist(filtered.patch(minalpha=0.1,slope=0.5,facecolor=[0,1,1]))
    title('Merge')    
    P.publish()

    ax.clear()

    return real,filtered 

#the actual, hidden state
real=numpy.array([[0,4]])

#the sensor
sensor=Mvar(vectors=[[1,0],[0,1]],var=[1,numpy.Inf])

#the system noise
noise=Mvar(vectors=[[1,-0.2],[0.2,1]],var=[0.25,1])

#the shear transform to move the system forward
transform=Matrix([[1,0],[1,1]])

filtered=sensor.measure(real)

P=publisher()

#create the figure and axis
F=figure()
ax=F.add_subplot(1,1,1,aspect='equal')

#plot the initial actual position
plot(real[:,0],real[:,1],'v',color=[1,0,1],ms=20)
title('Kalman Filtering: Start')
note=P.publish()

#measure the actual position, and plot the measurment
ax.add_artist(filtered.patch(minalpha=0.1,slope=0.5,facecolor=[0,0,1]))
title('Initialize to first measurment')
P.publish()
ax.clear()


(real,filtered) =do(real,filtered,sensor,noise,transform)
(real,filtered) =do(real,filtered,sensor,noise,transform)
(real,filtered) =do(real,filtered,sensor,noise,transform)
(real,filtered) =do(real,filtered,sensor,noise,transform)
(real,filtered) =do(real,filtered,sensor,noise,transform)
(real,filtered) =do(real,filtered,sensor,noise,transform)
