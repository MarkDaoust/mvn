#! /usr/bin/env python

import numpy
import collections
import itertools

def sample(item,count):
    if hasattr(item,'sample'):
        return item.sample(count)
    else:
        return list(itertools.repeat(item,count))

    
        

class Mixture(object):
    def __init__(self,items,weights):
        self.items = items
        self.weights = numpy.array(weights)
        
    def sample(self,shape):
        try:
            shape=list(shape)
        except TypeError:
            shape=[shape]

        shape.insert(0,1)

        rolls = numpy.random.rand(*shape)

        indexes = (rolls>=numpy.cumsum(self.weights)[:,None]).sum(0)

        print indexes

        counts = collections.Counter(indexes)
        print counts
        items = ((self.items[index],count) for (index,count) in counts.iteritems())
        samples = [sample(item,count) for (item,count) in items]
        samples = numpy.concatenate(samples)
        return numpy.random.shuffle(samples)

if __name__ == '__main__':
    M = Mixture([1,2,3],[0.2,0.3,0.5])
    print M.sample(20)
