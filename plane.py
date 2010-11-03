#! /usr/bin/env python

############  imports

## builtins
import itertools
import collections 
import copy
import operator

## 3rd party
import numpy
import scipy

## maybe imports: third party things that we can live without
from maybe import Ellipse

## local
import helpers

from square import square
from automath import Automath
from right import Right
from inplace import Inplace
from matrix import Matrix

class Plane(object,Automath,Right,Inplace):    
    ############## Creation
    def __init__(self,
        vectors=Matrix.eye,
        var=numpy.ones,
        mean=numpy.zeros,
        square=True,
        squeeze=True,
        **kwargs
    ):
        #stack everything to check sizes and automatically inflate any 
        #functions that were passed in
        
        var= var if callable(var) else numpy.array(var).flatten()[:,None]
        mean= mean if callable(mean) else numpy.array(mean).flatten()[None,:]
        vectors= vectors if callable(vectors) else Matrix(vectors)
        
        stack=Matrix(helpers.autostack([
            [var,vectors],
            [1  ,mean   ],
        ]))
        
        #unpack the stack into the object's parameters
        self.mean = numpy.real_if_close(stack[-1,1:])
        self.var = numpy.real_if_close(numpy.array(stack[:-1,0]).flatten())
        self.vectors = numpy.real_if_close(stack[:-1,1:])
        
        if square:
            self.copy(self.square())

        if squeeze:
            self.copy(self.squeeze(**kwargs))

    ############## alternate creation methods
    @staticmethod
    def fromCov(cov,**kwargs):
        """
        everything in kwargs is passed directly to the constructor
        """
        diag = Matrix(numpy.diag(cov))
        eig = numpy.linalg.eigh if abs(diag) == diag else numpy.linalg.eig
        #get the variances and vectors.
        (var,vectors) = eig(cov) if cov.size else (Matrix([]),Matrix([]))
        vectors=Matrix(vectors.H)     

        return Mvar(
            vectors=vectors,
            var=var,
            square=False,
            **kwargs
        )
    
    @staticmethod
    def fromData(data,mean=None,weights=None, bias=False, **kwargs):        
        if isinstance(data,Mvar):
            return data.copy()
        
        data=Matrix(data)

        
        #todo: implement these
        assert mean is None,'standard error not yet implemented'
        assert weights is None,'weights not implemented'
        assert data.dtype is not numpy.dtype('object'),'not mplementd for mvars yet'
        
        #get the number of samples, subtract 1 if un-biased
        N=data.shape[0] if bias else data.shape[0]-1
        
        #get the mean of the data
        mean=numpy.mean(data,axis=0)
        
        cov=(data.H*data)/N-mean.H*mean
        
        #create the mvar from the mean and covariance of the data
        return Mvar.fromCov(
            cov = cov,
            mean= mean,
            **kwargs
        )
        
    ##### 'cosmetic' manipulations
    def square(self):
        """
        squares up the vectors, so that the 'vectors' matrix is unitary 
        (rotation matrix extended to complex numbers)
        
        >>> assert A.vectors*A.vectors.H==Matrix.eye
        """ 
        result=self.copy()
        (result.var,result.vectors)=square(
            vectors=self.vectors,
            var=self.var,
        )
        return result

    
    ############ setters/getters -> properties
    
    cov = property(
        fget=lambda self:self.vectors.H*numpy.diagflat(self.var)*self.vectors, 
        fset=lambda self,cov:self.copy(
            Mvar.fromCov(
                mean=self.mean,
                cov=cov,
        )),
        doc="""
            get or set the covariance matrix used by the object
        
            >>> assert A.cov==A.vectors.H*numpy.diagflat(A.var)*A.vectors
            >>> assert A.scaled.H*A.scaled==abs(A).cov
        """
    )
    
    ndim=property(
        fget=lambda self:(self.mean.size),
        doc="""
            get the number of dimensions of the space the mvar exists in
            >>> assert A.ndim==A.mean.size
        """
    )
    
    shape=property(
        fget=lambda self:(self.vectors.shape),
        doc="""
            get the shape of the vectors,the first element is the number of 
            vectors, the second is their lengths: the number of dimensions of 
            the space they are embedded in
            
            >>> assert A.vectors.shape == A.shape
            >>> assert (A.var.size,A.mean.size)==A.shape
            >>> assert A.shape[1]==A.ndim
        """
    )

    ########## Utilities
        
    @staticmethod
    def stack(*mvars,**kwargs):
        #no 'square' is necessary here because the rotation matrixes are in 
        #entierly different dimensions
        return Mvar(
            #stack the means
            mean=numpy.concatenate(mvar.mean for mvar in mvars),
            #stack the vector diagonally
            vectors=diagstack(mvar.vectors for mvar in mvars),
            var=numpy.concatenate(mvar.var for var in mvars),
            **kwargs
        )
    
    ############## indexing
    
    def given(self,index,value):
        #convert the inputs
        value=Mvar.fromData(value,bias=True)
        
        #create the mean, for the new object,and set the values of interest
        mean=numpy.zeros([1,self.shape[0]])
        mean[0,index]=value.mean

        #create empty vectors for the new object
        vectors=numpy.zeros([
            value.shape[0]+(self.ndim-value.ndim),
            self.ndim
        ])
        vectors[0:value.shape[0],index]=value.vectors
        vectors[value.shape[0]:,~index]=numpy.eye(self.ndim-value.ndim)
        
        #create the variance for the new object
        var=numpy.zeros(vectors.shape[0])
        var[0:value.shape[0]]=value.var
        var[value.shape[0]:]=numpy.Inf

        #blend with the self
        return self & Mvar(
            var=var,
            mean=mean,
            vectors=vectors,
        ) 
        
    def __setitem__(self,index,value):
        self.copy(self.given(index,value))

    def __getitem__(self,*index):
        index=tuple(index)
        if len(index)==1:
            index=index+(slice(None),)
        else:
            assert (len(index) == 2,'Invalid Index, should have 1 or 2 elements')
        
        return Mvar(
            mean=self.mean[:,index[1]],
            vectors=self.vectors[index[0],index[1]],
            var=self.var[index[0]],
        )

    ############ Math

    def __eq__(self,other):
        #check the number of dimensions of the space
        assert (
            self.ndim == other.ndim,
            """if the objects have different numbers of dimensions, you're doing something wrong"""
        )

        self=self.squeeze()
        other=other.squeeze()

        Sshape=self.shape
        Oshape=other.shape

        #check the number of flat dimensions in each object
        if Sshape[0]-Sshape[1] != Oshape[0] - Oshape[1]:
            return False

        
        Sfinite=numpy.isfinite(self.var)
        Ofinite=numpy.isfinite(other.var)

        if Sfinite.sum() != Ofinite.sum():
            return False

        if Sfinite.all():
            return self.mean==other.mean and self.cov == other.cov
    
        #remove the infinite directions from the means
        SIvectors=self.vectors[~Sfinite]
        Smean=self.mean - self.mean*SIvectors.H*SIvectors 

        OIvectors=other.vectors[~Ofinite]
        Omean=other.mean - other.mean*OIvectors.H*OIvectors

        #compare what's left of the means   
        if Smean != Omean:
            return False
   
        H=lambda M:M.H*M

        SFvectors=self.vectors[Sfinite]
        SFvar=self.var[Sfinite]

        OFvectors = other.vectors[Ofinite]
        OFvar = other.var[Ofinite]

        cov=lambda vectors,var: vectors.H*numpy.diagflat(var)*vectors

        #compare the finite and infinite covariances 
        return (
            cov(SFvectors,SFvar) == cov(OFvectors,SFvar) and
            SIvectors.H*SIvectors == OIvectors.H*OIvectors
        )
        
    
    def __or__(self,other):
        """
        self | other
        I don't  know what this means yet
        """
        #todo: create a 'GMM' class so that | has real meaning
        return self+other-self&other
    

    def __and__(self,other):    
        #assuming the mvars are squared and squeezed 
        #if they both fill the space        
        if (
            self.shape[0] == self.shape[1] and 
            other.shape[0]==other.shape[1]
        ):
            #then this is a standard paralell operation
            return (self**(-1)+other**(-1))**(-1) 
        
        #otherwise there is more work to do
        
        #invert each object        
        Iself=self**-1
        Iother=other**-1

        Fself=numpy.isfinite(Iself.var)
        Fother=numpy.isfinite(Iother.var)

        #the object's null vectors will show up as having infinite 
        #variances in the inverted objects        

        Nself=Iself.vectors[~Fself,:]
        Nother=Iother.vectors[~Fother,:] 

        null=numpy.vstack([
            Nself,
            Nother,
        ])

        #get length of the component of the means along each null vector
        r=numpy.vstack([Nself*self.mean.H,Nother*other.mean.H])

        (s,v,d)=numpy.linalg.svd(null,full_matrices=False)

        nonZero = ~helpers.approx(v)

        s=s[:,nonZero]
        v=v[nonZero]
        d=d[nonZero,:]
        
        Dmean=(numpy.diagflat(v**-1)*s.H*r).H*d

        new=(Iself+Iother+Mvar(vectors=d,var=Matrix.infs))**(-1)

        #add in the components of the means along each null vector
        new.mean=new.mean+Dmean

        return new
        

        
    def __mul__(self,other):        
        other=self._mulConvert(other)
        return self._multipliers[type(other)](self,other) 
    
    def _scalarMul(self,scalar):
        return Mvar(
            mean= scalar*self.mean,
            var = scalar*self.var,
            vectors = self.vectors,
            square = not numpy.isreal(scalar),
        )

    def _matrixMul(self,matrix):
        return Mvar(
            mean=self.mean*matrix,
            var=self.var,
            vectors=self.vectors*matrix,
        )

    def _mvarMul(self,mvar):
        result = (self*mvar.transform()+mvar*self.transform())
        
        result.mean += (
            self.mean-self.mean*mvar.transform(0)+
            mvar.mean-mvar.mean*self.transform(0)
        )

        return result/2

    @staticmethod
    def _mulConvert(
        item,
        helper=lambda item: Matrix(item) if item.ndim else item
    ):
        return (
            item if 
            isinstance(item,Mvar) else 
            helper(numpy.array(item))
        )

    def __rmul__(
        self,
        other,
    ):
        (transform,other)= self._rmulConvert(other)
        return self._rmultipliers[type(other)](transform,other)

    def _rmulConvert(self,other,
        helper=lambda self,other:(
            self.transform() if other.ndim else self,
            Matrix(other) if other.ndim else other,
        )
    ):
        return helper(self,numpy.array(other))

    _rmultipliers={
        #if the left operand is a matrix, the mvar has been converted to
        #to a matrix -> use matrix multiply
        (Matrix):lambda self,other:other*self,
        #if the left operand is a constant use scalar multiply
        (numpy.ndarray):_scalarMul
    }


    
    def __add__(self,other):
        #todo: fix the crash generated, for flat objects by: 1/A-1/A (inf-inf == nan)

        other = other if isinstance(other,Mvar) else Mvar(mean=other)
        return Mvar(
            mean=self.mean+other.mean,
            vectors=numpy.vstack([self.vectors,other.vectors]),
            var = numpy.concatenate([self.var,other.var]),
        )
        
    ################# Non-Math python internals
    def __call__(self,locations):
         return numpy.exp(self.dist2(self,locations))/2/numpy.pi/scipy.sqrt(self.det(self))
 
    def __repr__(self):
        """
        print self
        str(self)
        repr(self)
        """
        return '\n'.join([
            'Mvar(',
            '    mean=',8*' '+self.mean.__repr__().replace('\n','\n'+8*' ')+',',
            '    var=',8*' '+self.var.__repr__().replace('\n','\n'+8*' ')+',',
            '    vectors=',8*' '+self.vectors.__repr__().replace('\n','\n'+8*' ')+',',
            ')',
        ])
        
    
    __str__=__repr__