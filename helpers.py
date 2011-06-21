import itertools
import copy
import operator

import numpy

#from scipy import sqrt()
def sqrt(data):
    """
    like scipy.sqrt without a scipy depandancy
    """
    data = numpy.asarray(data)
    if numpy.isreal(data).all() and (data>=0).all():
        return numpy.sqrt(data)
    return data**(0.5+0j)

def squeeze(vectors,var,**kwargs):
    small=approx(var,**kwargs)
    
    if small.size:
        var = var[~small]
        vectors = vectors[~small,:]
    
    return var,vectors

def mag2(vectors):
    return numpy.real_if_close(
        (
            numpy.array(vectors)*numpy.array(vectors.conjugate())
        ).sum(axis = vectors.ndim-1)
    )

def sign(self):
    """
    improved sign function:
        returns a similar array of unit length (possibly complex) numbers pointing in the same 
        direction as the input 
    """
    return numpy.divide(
        self,
        abs(self),
    )
 
def unit(self):
    return scipy.sqrt(numpy.array(self)/mag2(self))

def ascomplex(self):
    """
    return an array pointing to the same data, but interpreting it as a 
    different type
    """
    shape=self.shape
    duplicate=copy.copy(self)
    duplicate.dtype=complex
    duplicate.shape=shape[:-1]
    return duplicate

def diagstack(arrays):
    """
    stack matrixes diagonally
    1d arrays are interpreted as 1xN
    it's like numpy.diag but works with matrixes

    output is two dimensional

    type matches first input, if it is a numpy array or matrix, 
    otherwise this returns a numpy array
    """
    #make a nested matrix, with the inputs on the diagonal, and the numpy.zeros
    #function everywhere else
    E=numpy.eye(len(arrays),dtype=bool)
    result=numpy.zeros(E.shape,dtype=object)
    for index,array in zip(numpy.argwhere(E),arrays):
        index=tuple(index)
        result[index]=array
    
    result[~E]=numpy.zeros    
    
    return autostack(result)

def autostack(rows,default=0):
    """
    simplify matrix stacking
    vertically stack the results of horizontally stacking each row in rows, 
    with no need to explicitly declare the size of and callables
    
    >>> autostack([ 
    ...     [numpy.eye(3),numpy.zeros],
    ...     [  numpy.ones,          1],
    ... ])
    matrix([[ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 1.,  1.,  1.,  1.]])
    
    Callables are called with a shape tuple as the only argument. 
    The 'default' parameter controls the size when a row of column contains 
    only callables, 

    >>> autostack([
    ...     [[1,2,3]],
    ...     [numpy.ones]
    ... ],default=0)
    matrix([[ 1.,  2.,  3.]])
    
    >>> autostack([
    ...     [[1,2,3]],
    ...     [numpy.ones]
    ... ],default=4)
    matrix([[ 1.,  2.,  3.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])

    >>> autostack([
    ...     [                       [1,2,3],   1],
    ...     [lambda shape:numpy.eye(*shape),[[1]
    ...                                    , [1]]]
    ... ])
    matrix([[ 1.,  2.,  3.,  1.],
            [ 1.,  0.,  0.,  1.],
            [ 0.,  1.,  0.,  1.]])
    """
    #first make sure everything is a 2 dimensional array
    data = [
        [[numpy.array(item,ndmin=2),callable(item)] for item in row]
        for row in rows      
    ]

    #make a 2d numpy array out of the data blocks
    data = numpy.array([
        list(itertools.chain([[None,None]],row)) 
        for row in data
    ],dtype = object
    )[:,1:,:]

    rows = data[...,0]
    calls = numpy.asarray(data[...,1],dtype = bool)

    #store the shape of the data
    shape=data.shape

    #if anything is callable
    if calls.any():
        #make an array of shapes

        [heights,widths] = numpy.vectorize(lambda item:item.shape)(rows)

        heights[calls]= -1
        widths[calls]= -1

        maxheight=heights.max(1)
        maxwidth=widths.max(0)
        
        #replace the -1 with the default value
        maxheight[maxheight==-1] = default
        maxwidth[maxwidth==-1] = default

        for (down,right) in numpy.argwhere(calls):
            #call the callable with (height,width) as the only argument
            #and drop it into the data slot,
            rows[down,right] = numpy.matrix(rows[down,right][0,0](
                (maxheight[down],maxwidth[right])
            ))
        
    #do the stacking    
    return numpy.vstack([
        numpy.hstack(row) 
        for row in rows
    ])


def paralell(*items):
    """
    resistors in paralell, and thanks to 
    duck typing and operator overloading, this happens 
    to be exactly what we need for kalman sensor fusion. 
    """
    inverted=[item**(-1) for item in items]
    return sum(inverted[1:],inverted[0])**(-1)

def approx(a,other = None,atol=1e-12):
    """
    returns True where delta<atol 
    """
    if other is None :
        delta = numpy.abs(a)
    else:
        if callable(other):
            other=other(a.shape)
        delta = numpy.abs(a-other)

    return (delta<atol)

def dots(*args):
    """
    like numpy.dot but takes any number or arguments
    """
    assert len(args)>1
    return reduce(numpy.dot,args)

def sortrows(data,column=0):
    return data[numpy.argsort(data[:,column].flatten()),:] if data.size else data

def rotation2d(angle):
    return numpy.array([
        [ numpy.cos(angle),numpy.sin(angle)],
        [-numpy.sin(angle),numpy.cos(angle)],
    ])


