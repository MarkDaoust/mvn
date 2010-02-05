import numpy

from helpers import autostack

class lane(object):
    def __init__(self,
        mean=numpy.zeros,
        vectors=numpy.zeros,
        do_square=False,
        do_unit=False
    ):
        stack=autostack([
            [vectors],
            [mean   ],
        ])
        
        self.vectors=stack[:-1,:]
        self.mean=stack[-1,:]
        
        if do_square: self.do_square()
        if do_unit: self.do_unit()
         
    def do_square(self):
        for (n,vector) in enumerate(self.vectors):
            _rmcomponent(vectors[n+1,:],dir=vector)
            
    def do_unit(self):
        self.vectors=numpy.apply_along_aixs(
            lambda vector: vector/numpy.sqrt(numpy.sum(vector**2)),
            axis=1,
            arr=vectors,
        )
    
    def _rmcomponent(vectors,dir):
        dir=dir/numpy.sum(dir)

        vectors-=numpy.apply_along_aixs(
            lambda vector,onto: numpy.dot(vector,onto)*onto,        
            axis=1,
            arr=vectors,
            args=[onto],
        )
    