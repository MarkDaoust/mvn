#import nose.plugins.doctest
#
#class Doctest(nose.plugins.doctest.Doctest):
#    def makeTest(self, obj, parent):
#        """
#        Look for doctests in the given object, which will be a
#        function, method or class.
#        """
#        name = getattr(obj, '__name__', 'Unnammed %s' % type(obj))
#        module = getmodule(parent)
#        
#        extraglobs = getattr(module,'globs',dict)()
#        
#        doctests = self.finder.find(obj, module=module, name=name,extraglobs = extraglobs)
#        if doctests:
#            for test in doctests:
#                if len(test.examples) == 0:
#                    continue
#                yield DocTestCase(test, obj=obj, optionflags=self.optionflags,
#                                  result_var=self.doctest_result_var)
#
#nose.plugins.doctest.Doctest = Doctest

#builtin
import cPickle
import doctest
import copy

def getDocTests(module,fixture=None):
    if fixture is None:
        fixture = {}
    
    def setUp(test,fixture = fixture):
       test.globs.update(copy.deepcopy(fixture))
    
    testCases=doctest.DocTestSuite(module, setUp = setUp)
    return testCases
