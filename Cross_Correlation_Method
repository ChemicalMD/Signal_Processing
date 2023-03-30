def Cross_Correlate(Reference, Waveform):
    
    Reference_X = [round(float(t),10) for t,v in zip(Reference[:,0],Reference[:,1])]
    Reference_Y = norm([v for t,v in zip(Reference[:,0],Reference[:,1])])
    
    sr_ref = round((Reference_X[1] - Reference_X[0])**(-1))
    sr_signal = round((Waveform[1,0] - Waveform[0,0])**(-1))
    
    if sr_signal != sr_ref:

        Ref, Wave, sr = resample_to_match(Reference,Waveform)
    
        Ref_X = Ref[:,0]
        Ref_Y = Ref[:,1]
    
        X = [round(float(t),10) for t in Wave[:,0]]
        Y_short = norm(Wave[:,1])

        Y_ref = np.append(Ref_Y, [y for y in np.zeros(1 + int((max(X) - max(Ref_X))*sr))])

        Y_corr = np.append([y for y in np.zeros(int((min(X))*sr))], Y_short)

        corr = signal.correlate(Y_corr, Y_ref, mode="full")
        corr = np.insert(corr, 0, 0.0)
        corr = norm(corr)
    
        n = len(Y_corr)

        delay_array = np.linspace(-n/sr, n/sr, 2*n)
    
        arrival = (delay_array[np.argmax(corr)])
    
        return arrival, corr, delay_array, Ref, Wave
    
    else:
        
        Ref_X = Reference[:,0]
        Ref_Y = Reference[:,1]
        
        X = [round(float(t),10) for t in Waveform[:,0]]
        Y_short = norm(Waveform[:,1])
        
        Y_ref = np.append(Reference_Y, [y for y in np.zeros(1 + int((max(X) - max(Ref_X))*sr_ref))])

        Y_corr = np.append([y for y in np.zeros(int((min(X))*sr_ref))], Y_short)

        corr = signal.correlate(Y_corr, Y_ref, mode="full")
        corr = np.insert(corr, 0, 0.0)
        corr = norm(corr)
    
        n = len(Y_corr)

        delay_array = np.linspace(-n/sr_ref, n/sr_ref, 2*n)
    
        arrival = (delay_array[np.argmax(corr)])
    
        return arrival, corr, delay_array, Reference, Waveform
