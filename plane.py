import numpy

from helpers import autostack
from matrix import Matrix 

class Plane(object):
    """
    there's a problem here: this creates something who's transpose is it's unverse,
    not something who's
    """
    
    def __init__(self,
        mean=numpy.zeros,
        vectors=numpy.zeros,
        do_square=True,
        do_unit=True
    ):
        stack=Matrix(autostack([
            [numpy.zeros,vectors],
            [          1,mean   ],
        ]))
        
        self.vectors=stack[:-1,1:]
        self.mean=stack[-1,1:]
        
        if do_square: self.do_square()
        if do_unit: self.do_unit()
         
    def do_square(self):
        vectors=self.vectors
        for (n,vector) in enumerate(vectors[:-1,:]):
            vectors[n+1:,:]-=_project(vectors[n+1:,:],dir=vector)
        self.vectors=vectors
        
    def do_unit(self):
        vectors=self.vectors
        #the two .T's here are just to deal with numpy broadcasting, don't replace them with .T's
        self.vectors=Matrix(numpy.array(vectors.T)/numpy.sqrt(numpy.sum(numpy.array(vectors)**2,1))).T
        
        
def _project(vectors,dir):
    dir=dir/numpy.sqrt(numpy.sum(numpy.array(dir)**2))
    
    return numpy.apply_along_axis(
        lambda vector,onto=dir: numpy.dot(vector,onto.T),        
        axis=1,
        arr=vectors,
    )*dir

Q=Plane(vectors=Matrix([[1+0j,0+2j,3+4j],[2,1+3j,0+0j],[0+1j,2,0+5j]])).vectors