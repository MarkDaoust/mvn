#! /usr/bin/env python
"""
************
Test Objects
************
"""

import os
import sys
import copy
import pickle
import optparse

import numpy


from mvn import Mvn
from mvn.matrix import Matrix
from mvn.helpers import randint    

(fileDir, _) = os.path.split(os.path.abspath(__file__))

pickleName =os.path.join(fileDir,'testObjects.pkl')

lookup ={}
try:
    lookup =pickle.load(open(pickleName,"r"))
    locals().update(lookup['last'])
except EOFError:
    pass
except IOError: 
    pass
except ValueError:
    pass
except KeyError:
    pass

    
def parse(argv):
    #make parser
    parser=optparse.OptionParser()

    new=optparse.OptionGroup(parser,'new')    
    new.add_option('-n','--new',action='store_true',default=False,help='force creation of new test objects')
    parser.add_option_group(new)

    general=optparse.OptionGroup(parser,'general')
    general.add_option('-s','--seed',action='store',type=int,help='set the random seed')
    general.add_option('-d','--ndim',action='store',type=int,help='set the number of data dimensions')
    parser.add_option_group(general)

    flatness=optparse.OptionGroup(parser,'Flatness')
    flatness.add_option('-f','--flat',dest='flat',action='store_const',const=True,help='set all the objects to flat')
    flatness.add_option('-F','--full',dest='flat',action='store_const',const=False,help='set no objects to flat')
    flatness.add_option('--flatness',nargs=3,dest='flat',type=int,help='set flatness individually')
    parser.add_option_group(flatness)

    #parse
    (settings,remainder) = parser.parse_args(argv)
    assert not remainder
    return settings    

class MvnFixture(object):
    def __init__(self,args):
        if isinstance(args,dict):
            fixDict = args
        else:
            settings = parse(args)
            fixDict = self.getObjects(settings)
            
        object.__setattr__(self,'contents',{})
        
        #this manual unpacking is for pylint, 
        #    so it can see what is in the test fixture
        self.ndim = fixDict['ndim'] 
        self.A = fixDict['A']
        self.B =fixDict['B']
        self.C = fixDict['C']
        self.M = fixDict['M']
        self.M2 = fixDict['M2']
        self.E = fixDict['E']
        self.K1 = fixDict['K1']
        self.K2 = fixDict['K2']
        self.N = fixDict['N']
 
           
    @staticmethod
    def getObjects(values):
        drop = ['new','seed']
        frozenValues=frozenset(
            (key,value) 
            for (key,value) in values.__dict__.iteritems() 
            if key not in drop
        )
    
        objects=None
        if not values.new:
            try:
                objects=lookup[frozenValues]
            except KeyError:
                pass
    
        if objects is None:
            objects = MvnFixture.makeObjects(
                values.flat,
                values.ndim,
                values.seed
            )
            
        lookup[frozenValues] = objects
    
        lookup['last']=objects
        globals().update(objects)
    
        pickle.dump(lookup,open(pickleName,'w'))
    
        return objects
    
    @staticmethod
    def makeObjects(flat=None,ndim=None,seed=None):        
        if seed is None:
            seed=randint(1,1e6)
            
        assert isinstance(seed,int),'seed must be an int'

        numpy.random.seed(seed)
        randn=numpy.random.randn
    
        if ndim is None:
            ndim=randint(0,20)
            
        assert isinstance(ndim,int),'ndim must be an int'
    
        shapes={
            None:lambda :max(randint(-ndim,ndim),0),
            True:lambda :randint(1,ndim),
            False:lambda :0,
        }
    
        triple=lambda x:[x,x,x]
        
        if flat in shapes:
            flat=[item() for item in triple(shapes[flat])]
        elif isinstance(flat,int):
            flat=triple(flat)
        
        assert all(f<=ndim for f in flat), "flatness can't be larger than ndim"
        
        rvec= lambda n=1,ndim=ndim:Matrix(randn(n,ndim))
                
        A,B,C=[
            Mvn.rand([ndim-F,ndim])
            for F in flat
        ]
    
    
        n=randint(1,2*ndim)
        M=rvec(n).H
        M2=rvec(n).H    
    
        E=Matrix.eye(ndim)
        
        K1 = (numpy.random.randn())
        K2 = (numpy.random.randn())
    
        N=randint(-5,5)
        
        return {
            'ndim' : ndim,
            'A' : A,
            'B' : B,
            'C' : C,
            'M' : M,
            'M2' : M2,
            'E' : E,
            'K1' : K1,
            'K2' : K2,
            'N' : N,
        }

        
    def __setattr__(self,name,value):
        assert name not in self.contents
        
        object.__setattr__(self,name,value)
        self.contents[name] = value
        
    def reset(self):
        self.__dict__.update(copy.deepcopy(self.contents))        
            

    
if __name__ == '__main__':
    args = sys.argv[1:]
    print MvnFixture(args).contents
    
