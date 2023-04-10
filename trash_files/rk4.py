########################################################################
## 4th order Runge-Kutta
########################################################################

def rk4(fn, xin, uin, t):
    """
    4th order Runge-Kutta method for solving ODEs
    
    Parameters
    ----------
    fn : function
        function handle for the system dynamics equation f(x,u,t)

    xin : numpy.ndarray
        initial state vector
    
    uin : numpy.ndarray
        input vector

    t : numpy.ndarray
        time vector

    Returns
    -------
    t : numpy.ndarray
        time vector (same as input)

    x : numpy.ndarray
        state vector
        """
    
    a   = t[0]
    b   = t[1]
    w   = xin
    N   = 2
    h   = (b - a)/N
    t   = a

    for j in range(1, N+1):
        K1  = h*fn(t, w, uin)
        K2  = h*fn(t+h/2, w+K1/2, uin)
        K3  = h*fn(t+h/2, w+K2/2, uin)
        K4  = h*fn(t+h, w+K3, uin)
        
        w   = w + (K1 + 2*K2 + 2*K3 +K4)/6
        t   = a+j*h
    
    return t, w
        
        