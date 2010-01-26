import numpy

def astype(self,newtype):
    duplicate=self
    duplicate.dtype=newtype
    return duplicate

def diagstack(arrays):
    """
    output is two dimensional

    type matches first input, if it is a numpy array or matrix, 
    otherwise this returns a numpy array
    """
    #make a nested matrix, with the inputs on the diagonal, and the numpy.zeros
    #function everywhere else
    arrays=numpy.where(
        eye,
        numpy.diag(numpy.array(arrays,dtype=object)),
        numpy.zeros,
    )
    
    return autostack(result)

def autostack(rows,default=0):
    """
    simplify matrix stacking
    vertically stack the results of horizontally stacking each row in rows, 
    with no need to explicitly declare the size of the zeros
    
    autostack([ 
        [std,numpy.zeros]
        [mean,1]
    ])
    
    rows are horizontally stacked first, then vertically stacked, so the cells 
    do not need to line up vertically they just need the same total size
    
    if there are callables in the data things must align vertically as well 

    callables are called with a shape tuple as the only argument. or zeros 
    where a row or column contains only callables.
    """
    #convert the data items into an object array 
    data=numpy.array(rows,dtype = object)
    
    #store the shape
    shape=data.shape

    #and the type of the first numpy thing
    types=[type(item) for item in data.flat if isinstance(item,numpy.ndarray)]
    atype=numpy.matrix if (types and types[0] is numpy.matrix) else numpy.array
    
    #make a matrix of the callable status of each item
    calls=numpy.array([callable(item) for item in data.flat]).reshape(shape)
 
    #everythere the data is not callable convert to a matrix
    data[~calls]=[
        numpy.matrix(item) for (call,item) 
        in zip(calls.flat,data.flat) 
        if ~call
    ]
    
    if calls.any():
        sizes = numpy.array([
            (default,default) if call else item.shape 
            for call,item in zip(calls.flat,data.flat)
        ]).reshape(shape+(2,))
        
        heights=sizes[...,0]
        widths=sizes[...,1]
        
        for (down,right) in numpy.argwhere(calls):
            #call the callable with (height,width) as the only argument
            #and drop it into the data slot,max is used because callables 
            #default to zero, so you don't pass in a zero shape uness the 
            #row/column only contains callables
            data[down,right] = numpy.matrix(data[down,right](
                (heights[down,:].max(),widths[:,right].max())
            ))
            
    return atype(numpy.vstack([
        numpy.hstack(row) 
        for row in data
    ]))


def paralell(*items):
    """
    resistors in paralell, and thanks to 
    duck typing and operator overloading, this happens 
    to be exactly what we need for kalman sensor fusion. 
    """
    inverted=[item**(-1) for item in items]
    return sum(inverted[1:],inverted[0])**(-1)

def close(a,other = None,atol=1e-5,rtol=1e-8):
    if other is not None :
        delta = numpy.abs(a-other)
    else:
        delta = numpy.abs(a)
    
    MAX = numpy.max(delta)
    if MAX<atol:
        return numpy.zeros(delta.shape,dtype=bool)
    
    return ~((delta>atol) & (delta/MAX>rtol))

def rotation2d(angle):
    return numpy.array([
        [ numpy.cos(angle),numpy.sin(angle)],
        [-numpy.sin(angle),numpy.cos(angle)],
    ])