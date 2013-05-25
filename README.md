Multivariate Normal Distributions, in Python/Numpy.
=============================================

    import mvn

Example application: <http://en.wikipedia.org/wiki/Talk:Kalman_filter#Example_Animation>
![Animation](http://upload.wikimedia.org/wikipedia/commons/5/5e/Kalman_filter_animation%2C_1d.gif)

### WARNING:

    1) I'm learning all of this as I do it, I may have made some mistakes.
        If it doesn't have automated tests, it doesn't work.

    2) Second system effect: what's here currently is the "first-system".
        Expect everything to change

## Target API     
The goal is to make these probability distributions 'easy'. 
Not all of this works yet

### Principal Components:

    M = mvn.Mvn(data)
    P = M(1:3)

### Sensor fusion

    result = sensor1 & sensor2 & sensor3

### Bayseian filtering:
    
    # Linear Kalman Filter
    state[t+1] = (state[t]*stm + noise) & measurment
    
    # Unscented Kalman Filter 
    state[t+1] = mvn.Mvn(stateupdate(state[t].simplex())) & measurement

    # Particle Filter 
    # (states are mixtures of points)
    state[t+1] = stateupdate(state[t]) & measurement

### Expectation Maximization:

    mix = Mixture([A,B,C])
    mix = mix.fit(data)

### Regression & uncertainty:

    M = mvn.Mvn(data)
    y = regression.given(x = 10)

### Statistics:

    dist = mvn.Mvn(data)
    dist.mean == data.mean
    dist.cov == data.cov

    quadrant = (dist > 0).all()
    dist.min()
    dist.max()

### Plotting (matplotlib):

    mvn.Mvn(data).plot()
