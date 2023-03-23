def Analyze(X,Y):

    T = [x for x,y in zip(X,Y) if 9.6e-6 <= x <= 13e-6]
    V = [y for x,y in zip(X,Y) if 9.6e-6 <= x <= 13e-6]

    T = [x for x,y in zip(X,Y) if 9.6e-6 <= x <= T[V.index(min(V))-1]]
    V = [y for x,y in zip(X,Y) if 9.6e-6 <= x <= T[V.index(min(V))-1]]

    T_noise = [x for x,y in zip(X,Y) if 9.5e-6 <= x <= 10.5e-6]
    V_noise = [y for x,y in zip(X,Y) if 9.5e-6 <= x <= 10.5e-6]

    global derivative
    derivative = np.gradient(V,T)
    derivative_noise = np.gradient(V_noise,T_noise)

    mean_noise = np.mean(V_noise)

    mean = np.mean(derivative_noise)
    std = np.std(derivative_noise)

    for t,v,d in zip(T,V,derivative):

        if np.abs(d) > np.abs(mean + 10*np.abs(std)):

            descent = t
            break

        else:

            continue

    T = [x for x,y in zip(X,Y) if descent <= x <= (descent + 0.2e-6)]
    V = [y for x,y in zip(X,Y) if descent <= x <= (descent + 0.2e-6)]

    T_descent = [x for x,y in zip(X,Y) if descent <= x <= T[V.index(min(V))-1]]
    V_descent = [y for x,y in zip(X,Y) if descent <= x <= T[V.index(min(V))-1]]

    m, b = np.polyfit(T_descent, V_descent, deg=1)

    arrival_P = (mean_noise - b) / m

    T_S = [x for x,y in zip(X,Y) if 15.5e-6 <= x <= 19e-6]
    V_S = [y for x,y in zip(X,Y) if 15.5e-6 <= x <= 19e-6]

    T_S = [x for x,y in zip(X,Y) if 15.5e-6 <= x <= (T_S[V_S.index(max(V_S))] + 0.3e-6)]
    V_S = [y for x,y in zip(X,Y) if 15.5e-6 <= x <= (T_S[V_S.index(max(V_S))] + 0.3e-6)]

    arrival_S = T_S[V_S.index(min(V_S))]

    return m, b, mean_noise, descent, arrival_P, arrival_S
