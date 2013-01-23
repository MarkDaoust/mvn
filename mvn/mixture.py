#! /usr/bin/env python
"""
*************
Mixture Class
*************
"""
import pylab
import numpy
import collections
import itertools

def sample(item, count):
    if hasattr(item, 'sample'):
        return item.sample(count)
    else:
        return list(itertools.repeat(item, count))    

class Mixture(object):
    def __init__(self, items, weights = None):
        items = list(items)
        
        if weights is None:
            weights = numpy.ones(len(items), float)
        else:
            weights = numpy.array(weights)
        
        weights = weights/weights.sum()
        
        assert len(weights) == len(items)
        
        self.items = items
        self.weights = weights
        
    def sample(self, shape):
        shape = numpy.array(shape, ndmin = 1)

        rolls = numpy.random.rand(*shape)

        indexes = (rolls >= numpy.cumsum(self.weights)[:, None]).sum(0)

        counts = collections.Counter(indexes)
        
        items = ((self.items[index], count) for (index, count) in counts.iteritems())
        samples = [sample(item, count) for (item, count) in items]

        samples = numpy.concatenate(samples)
        
        numpy.random.shuffle(samples)
        return samples

    def plot(self, ax = None, alpha = 'auto', **kwargs):
        if alpha == 'auto':
            alphas = self.weights/self.weights.max()
        elif hasattr(alpha, '__iter__'):
            alphas = list(alphas)
        else:
            alphas = [alpha for item in self.items]
        
        for item, alpha in zip(self.items, alphas):
            if hasattr(item, 'plot'):
                item.plot(ax,alpha = alpha,**kwargs)
            else:
                if ax is None:
                    ax = pylab.gca()
                ax.plot(item, alpha = alpha)
            
    def fit(self, data, weights = None):
        
        dataWeights = numpy.array([
            item.density(data) 
            for item in self.items
        ]).T
        
        if weights is not None:           
            dataWeights = dataWeights*weights[:, None]

        itemTotals = dataWeights.sum(0)
        newWeights = itemTotals/itemTotals.sum()
        
        dataTotals = dataWeights.sum(1)

        dataMembership = dataWeights/dataTotals[:, None]   
        
        newItems = [
            item.fit(
                data = data,
                weights = dataMembership[:, n],
                bias = True
            )
            for n, item in enumerate(self.items)
        ]          
        
        return Mixture(newItems, newWeights)
        
    def __getitem__(self, index):
        return Mixture(
            [item[index] for item in self.items],
            self.weights,        
        )
        
    def __len__(self):
        assert len(self.items) == len(self.weights)
        return len(self.items)
        
        
        
    def fit2(self, data, weights = None):
        dataWeights = numpy.array([
            item.density(data) 
            for item in self.items
        ]).T
        
        if weights is None:           
            dataWeights = dataWeights*weights[:, None]

        itemTotals = dataWeights.sum(0)
        self.weights = itemTotals/itemTotals.sum()
        
        dataTotals = dataWeights.sum(1)

        dataMembership = dataWeights/dataTotals[:, None]        
    
        self.items = [
            item.fit(
                data = data,
                weights = dataMembership,
                bias = True
            )
            for n, item in enumerate(self.items)
        ]
        
if __name__ == '__main__':
    M = Mixture([1, 2, 'asd', None], [0.2, 0.3, 0.5, 0.25])
    print M.sample(20)
