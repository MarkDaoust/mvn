from mvn import Mvn

L1=Mvn(mean=[0,0],vectors=[[1,1],[1,-1]], var=[numpy.inf,0.5])
L2=Mvn(mean=[1,0],vectors=[0,1],var=numpy.inf) 

L1 & L2

L1.given(dims=0,value=1) == L1&L2 
(L1&L2).mean==[1,1] 
(L1&L2).cov==[[0,0],[0,2]] 
    