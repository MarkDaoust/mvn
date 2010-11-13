#! /usr/bin/env python

from pylab import *

from mvar import Mvar

A=Mvar(mean=[0,3],vectors=[[2,1],[-0.75,0.75]])
B=Mvar(mean=[1,0],vectors=[[-2,1],[0.7,0.1]])

a=A.patch()
b=B.patch(facecolor='r')

AB=A&B

ab=AB.patch(facecolor='m')


F=figure()
ax=subplot(1,1,1)
axis('equal')
xlim([-5,5])
ylim([-5,5])


ax.add_artist(a)
ax.add_artist(b)
ax.add_artist(ab)

show()