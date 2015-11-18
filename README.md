Multivariate Normal Distributions, in Python/Numpy.
=============================================

    import mvn

Example application: <http://en.wikipedia.org/wiki/Talk:Kalman_filter#Example_Animation>
![Animation](http://upload.wikimedia.org/wikipedia/commons/5/5e/Kalman_filter_animation%2C_1d.gif)

### WARNING:

1. I was learning all of this as I did it, I may have made some mistakes.
2. Often I was more interested in whether *I could* than whether *I should*.
3. If it doesn't have automated tests, it probably doesn't work.


## Target API     
The goal is to make these probability distributions 'easy'. 
Not all of this works yet. 

### Sensor fusion

    result = sensor1 & sensor2 & sensor3

### Bayseian filtering:
    
    # Linear Kalman Filter
    state[t+1] = (state[t]*stm + noise) & measurment
    
    # Unscented Kalman Filter 
    state[t+1] = mvn.Mvn(stateupdate(state[t].simplex())) & measurement

    # Particle Filter 
    # (states are mixtures of points)
    state[t+1] = (stateupdate(state[t]) & measurement).resample()

### Expectation Maximization:

    mix = Mixture([A,B,C])
    mix = mix.fit(data)

### Regression & uncertainty:

    M = mvn.Mvn(data)
    M[0] = 10

### Statistics:

    dist = mvn.Mvn(data)
    dist.mean == data.mean()
    dist.cov == data.cov()

### Projection

    T = np.Matrix(...)
    dist = mvn.Mvn(data)
    dist*T == mvn.Mvn(data*T)

    dist2d = dist[:2]

### Integration
    
    p = dist.inBox(corner_1,corner_2)

### Plotting (matplotlib):

    mvn.Mvn(data).plot()
