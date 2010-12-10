Traceback (most recent call last):
  File "./runTests.py", line 45, in <module>
    jars=makejars()
  File "./runTests.py", line 18, in makejars
    objects=testTools.makeObjects(cplx=cplx,flat=flat)  
  File "/home/olpc/personal-projects/kalman/testTools.py", line 20, in makeObjects
    num=randint(1,ndim-1)
  File "mtrand.pyx", line 617, in mtrand.RandomState.randint
ValueError: low >= high
