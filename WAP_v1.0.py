# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 22:50:38 2022

@author: jeogb
"""

import PySimpleGUI as sg
from pathlib import Path
import re
import numpy as np
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_pdf
from matplotlib import use as use_agg
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if True: # imports and plotting style
    
    import matplotlib
    
    style = {
        "figure.figsize": (12, 8),
        "font.size": 7,
        "axes.labelsize": "16",
        "axes.titlesize": "16",
        "xtick.labelsize": "16",
        "ytick.labelsize": "16",
        "legend.fontsize": "18",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 2,
        "lines.markersize": 12,
        "xtick.major.size": 18,
        "ytick.major.size": 18,
        "xtick.top": True,
        "ytick.right": True,
    }
    
    matplotlib.rcParams.update(style)
    markers = ["o", "X",  "^", "P", "d", "*", "s", ".", "x", ">"] * 5
    matplotlib.rcParams["font.family"] = ["serif"]

#============================================================================#

def pack_figure(graph, figure):
    canvas = FigureCanvasTkAgg(figure, graph.Widget)
    plot_widget = canvas.get_tk_widget()
    plot_widget.pack(side='top', fill='both', expand=1)
    return plot_widget

#============================================================================#

def plot_waveform(X,Y):
    #fig = plt.figure(1)
    ax = plt.gca() # get current axes (figure)
    ax.cla() # clear the current figure
    ax.set_title('Waveform')
    ax.set_xlabel(r'Time ($\mu$s)')
    ax.set_ylabel('Normalized Amplitude')
    ax.plot(X, Y, color='blue', linestyle='-')
    canvas.draw()

#============================================================================#

def norm(signal):
  # Calculate the mean of the signal
  mean = sum(signal) / len(signal)
  
  # Subtract the mean from each point in the signal to remove the DC offset
  signal = [point - mean for point in signal]
  
  # Calculate the maximum and minimum values of the signal
  max_value = max(signal)
  min_value = min(signal)
  
  # Normalize the signal by dividing each point by the range of the signal and multiplying by 2
  range = max_value - min_value
  signal = [2 * (point / range) for point in signal]
  
  return signal

#============================================================================#

def resample_to_match(waveform1, waveform2):
    
  # Extract the time and sample values from the two waveforms
  time1 = waveform1[:,0]
  samples1 = waveform1[:,1]
  
  time2 = waveform2[:,0]
  samples2 = waveform2[:,1]

  # Calculate the time per sample for each waveform
  time_per_sample1 = time1[1] - time1[0]
  time_per_sample2 = time2[1] - time2[0]
  
  # Calculate the sampling rate of each waveform
  sampling_rate1 = 1.0 / time_per_sample1
  sampling_rate2 = 1.0 / time_per_sample2
  
  # Check which waveform has the higher sampling rate
  if 1.0 / time_per_sample1 > 1.0 / time_per_sample2:
      
    # Waveform 1 has a higher sampling rate, so resample it to match waveform 2
    time_per_sample = time_per_sample2
    num_samples = len(samples1)
    duration = time1[-1] - time1[0]
    new_num_samples = math.ceil(duration * 1.0 / time_per_sample)
    
    t = [i * time_per_sample for i in range(new_num_samples)]
    sample_index = [int(i * time_per_sample / time_per_sample1) for i in range(new_num_samples)]
    samples = samples1[sample_index]
    new_waveform = np.transpose(np.array((t,samples)))
    
    return new_waveform, waveform2, sampling_rate2

  else:
    # Waveform 2 has a higher or equal sampling rate, so resample it to match waveform 1
    time_per_sample = time_per_sample1
    num_samples = len(samples2)
    duration = time2[-1] - time2[0]
    new_num_samples = math.ceil(duration * 1.0 / time_per_sample)
    
    t = [i * time_per_sample for i in range(new_num_samples)]
    sample_index = [int(i * time_per_sample / time_per_sample2) for i in range(new_num_samples)]
    samples = samples2[sample_index]
    new_waveform = np.transpose(np.array((t,samples)))
    
    return waveform1, new_waveform, sampling_rate1

#============================================================================#

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

#============================================================================#

def overlap(Reference, Waveform, arrival):
    
    Reference_X = [round(float(t),10) for t,v in zip(Reference[:,0],Reference[:,1])]
    Reference_Y = norm([v for t,v in zip(Reference[:,0],Reference[:,1])])
    
    X = Waveform[:,0]
    Y = norm(Waveform[:,1])
    
    ax = plt.gca() # get current axes (figure)
    ax.cla() # clear the current figure
    plt.plot([(t+arrival)*1e6 for t in Reference_X], norm(Reference_Y), color="blue", linestyle="-", label="No Sample")
    plt.plot([t*1e6 for t in X], norm(Y), color="red", linestyle="-", label="Sample")
    plt.axvline(x=arrival*1e6, linestyle="--", color="k", alpha=0.75, label="Arrival Time = " + str(np.round((arrival*1e6), 3)) + r" $\mu$s")
    ax.set_xlabel(r"Time Shift ($\mu$s)")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_title('Signal Overlap')
    ax.legend(loc="best")
    fig.canvas.draw()

#============================================================================#

class Canvas(FigureCanvasTkAgg):
    """
    Create a canvas for matplotlib pyplot under tkinter/PySimpleGUI canvas
    """
    def __init__(self, figure=None, master=None):
        super().__init__(figure=figure, master=master)
        self.canvas = self.get_tk_widget()
        self.canvas.pack(side='top', fill='both', expand=1)

if True: # layout and window
    
    sg.theme('GrayGrayGray')
    
    use_agg('TkAgg')
    
    controls_col = sg.Column([
        [sg.Button('Select File', key='-OPENDATA-')],
        [sg.Text('File path:', key='-PATH-')],
        [sg.Text('File name:', key='-NAME-')],
        [sg.Button('Select Reference', key='-OPENREF-')],
        [sg.Text('File path:', key='-PATHREF-')],
        [sg.Text('File name:', key='-NAMEREF-')],
        [sg.Column([[sg.Frame('Read which columns? (X,Y)', layout=[
                    [sg.Text('('), sg.Input(size=(5,1), key='-XCOL-'), sg.Text(','), sg.Input(size=(5,1), key='-YCOL-'), sg.Text(')')],
                    ])]]),
        sg.Column([[sg.Frame('Skip how many rows?', layout=[
                    [sg.Input(size=(5,1), key='-SKIPROWS-')],
                    ])]])],
        [sg.Button('Plot Measurement', key='-PLOTDATA-'),
         sg.Button('Plot Reference', key='-PLOTREF-'),
         sg.Button('Plot Correlation', key='-CORR-'),
         sg.Button('Plot Overlap', key='-OVERLAP-')],
        [sg.Column([[sg.Frame('X-Limits (low,high)', layout=[
                    [sg.Text('('), sg.Input(size=(5,1), key='-XLOW-'), sg.Text(','), sg.Input(size=(5,1), key='-XHIGH-'), sg.Text(')')],
                    ])]]),
        sg.Column([[sg.Frame('Y-Limits (low,high)', layout=[
                    [sg.Text('('), sg.Input(size=(5,1), key='-YLOW-'), sg.Text(','), sg.Input(size=(5,1), key='-YHIGH-'), sg.Text(')')],
                    ])]])],
        [sg.Text('Save: '), sg.Spin(['.pdf','.png'], key='-SAVETYPE-'), sg.Button('Save', key='-SAVE-')],
        [sg.Frame('Messages', expand_x=True, expand_y=True, layout=[[sg.Text('(Messages will appear here)', key='-MESSAGE-')]])]
        ], expand_y=True)
    
    figure_col = sg.Frame("", [[sg.Canvas(background_color='green', expand_x=True, expand_y=True, key='-CANVAS-')]], size=(860, 645))
    
    layout = [
        [controls_col, figure_col]
        ]
    
    window = sg.Window("WAP_v1", layout, finalize=True)
    
    # Initialize graph
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    canvas = Canvas(fig, window['-CANVAS-'].Widget)
    ax.set_title('Waveform')
    ax.set_xlabel(r'Time ($\mu$s)')
    ax.set_ylabel('Normalized Amplitude')
    canvas.draw()
    
    have_data = False
    have_ref = False
    have_corr = False

#============================================================================#

while True:
    
    event, values = window.read()
    
    if event == sg.WIN_CLOSED:
        break
    
    if event == '-OPENDATA-':
        data_path = sg.popup_get_file('open', no_window=True)
        data_path_no_name = data_path.split('/')[:-1]
        window['-PATH-'].update('File path: ' + data_path + '/')
        
        if re.split(r'[/,.]', data_path)[-1] in ['csv','txt']:
            
            try:
                window['-NAME-'].update('File name: ' + data_path.split('/')[-1])
                have_data = True
                
            except Exception as e:
                window['-MESSAGE-'].update('ERROR: ' + str(e))
        
        else:
            window['-NAME-'].update('ERROR: wrong file type (' + data_path.split('/')[-1] + '), select .csv or .txt')
    
    if event == '-PLOTDATA-':
        if have_data:
            try:
                if (values['-XCOL-'] != '') & (values['-YCOL-'] != '') & (values['-SKIPROWS-'] != ''):
                    Data = np.loadtxt(data_path, delimiter=",", usecols=[int(values['-XCOL-']),int(values['-YCOL-'])], skiprows=int(values['-SKIPROWS-']))
                elif (values['-XCOL-'] != '') & (values['-YCOL-'] != ''):
                    Data = np.loadtxt(data_path, delimiter=",", usecols=[int(values['-XCOL-']),int(values['-YCOL-'])], skiprows=0)
                elif values['-SKIPROWS-'] != '':
                    Data = np.loadtxt(data_path, delimiter=",", usecols=[3,4], skiprows=int(values['-SKIPROWS-']))
                else:
                    Data = np.loadtxt(data_path, delimiter=",", usecols=[3,4], skiprows=0)
            except Exception as e:
                window['-NAME-'].update('ERROR: ' + str(e) + ' (file: ' + data_path.split('/')[-1] + ')')
            
            X = [t*1e6 for t in Data[:,0]]
            Y = norm(Data[:,1])
            
            ax.cla() # clear the current figure
            ax.plot(X, Y, color="blue", linestyle="-")
            
            if (values['-XLOW-'] != '') & (values['-XHIGH-'] != ''):
                ax.set_xlim(float(values['-XLOW-'])*1e6,float(values['-XHIGH-'])*1e6)
            if (values['-YLOW-'] != '') & (values['-YHIGH-'] != ''):
                ax.set_ylim(float(values['-YLOW-']),float(values['-YHIGH-']))
            
            ax.set_xlabel(r"Time ($\mu$s)")
            ax.set_ylabel('Normalized Amplitude')
            ax.set_title('Measurement')
            fig.tight_layout()
            canvas.draw()
    
    if event == '-OPENREF-':
        ref_path = sg.popup_get_file('open', no_window=True)
        ref_path_no_name = ref_path.split('/')[:-1]
        window['-PATHREF-'].update('File path: ' + ref_path + '/')
        
        if re.split(r'[/,.]', ref_path)[-1] in ['csv','txt']:
            
            try:
                window['-NAMEREF-'].update('File name: ' + ref_path.split('/')[-1])
                have_ref = True
                
            except Exception as e:
                window['-NAMEREF-'].update('ERROR: ' + str(e) + ' (file: ' + ref_path.split('/')[-1] + ')')
        
        else:
            window['-NAMEREF-'].update('ERROR: wrong file type (' + ref_path.split('/')[-1] + '), select .csv or .txt')
    
    if event == '-PLOTREF-':
        if have_ref:
            Ref_Data = np.loadtxt(ref_path, delimiter=",", usecols=[3,4], skiprows=0)
            X = [t*1e6 for t in Ref_Data[:,0]]
            Y = norm(Ref_Data[:,1])

            ax.cla() # clear the current figure
            ax.plot(X, Y, color="blue", linestyle="-")
            
            if (values['-XLOW-'] != '') & (values['-XHIGH-'] != ''):
                ax.set_xlim(float(values['-XLOW-'])*1e6,float(values['-XHIGH-'])*1e6)
            if (values['-YLOW-'] != '') & (values['-YHIGH-'] != ''):
                ax.set_ylim(float(values['-YLOW-']),float(values['-YHIGH-']))
            
            ax.set_xlabel(r"Time ($\mu$s)")
            ax.set_ylabel('Normalized Amplitude')
            ax.set_title('Reference')
            fig.tight_layout()
            canvas.draw()
    
    if event == '-CORR-':
        if have_data & have_ref:
            Ref_Data = np.loadtxt(ref_path, delimiter=",", usecols=[3,4], skiprows=0)
            Data = np.loadtxt(data_path, delimiter=",", usecols=[3,4], skiprows=0)
            arrival, corr, delay_array, Reference, Waveform = Cross_Correlate(Ref_Data, Data)
            if len(delay_array) < len(corr):
                delay_array = np.append(np.zeros(len(corr) - len(delay_array)), delay_array)

            ax.cla() # clear the current figure
            ax.plot([t*1e6 for t in delay_array], norm(corr), color="blue", linestyle="-")
            
            if (values['-XLOW-'] != '') & (values['-XHIGH-'] != ''):
                ax.set_xlim(float(values['-XLOW-']),float(values['-XHIGH-']))
            if (values['-YLOW-'] != '') & (values['-YHIGH-'] != ''):
                ax.set_ylim(float(values['-YLOW-']),float(values['-YHIGH-']))
            
            ax.set_xlabel(r"Time Shift ($\mu$s)")
            ax.set_ylabel("Normalized Amplitude")
            ax.set_title('Cross Correlation')
            fig.tight_layout()
            canvas.draw()
            
            have_corr = True
    
    if event == '-OVERLAP-':
        if have_corr:
            #overlap(Reference, Waveform, arrival)
            
            Reference_X = [round(float(t),10) for t,v in zip(Reference[:,0],Reference[:,1])]
            Reference_Y = norm([v for t,v in zip(Reference[:,0],Reference[:,1])])
            
            X = Waveform[:,0]
            Y = norm(Waveform[:,1])

            ax.cla() # clear the current figure
            ax.plot([(t+arrival)*1e6 for t in Reference_X], norm(Reference_Y), color="blue", linestyle="-", label="No Sample")
            ax.plot([t*1e6 for t in X], norm(Y), color="red", linestyle="-", label="Sample")
            y_min, y_max = ax.get_ylim()
            ax.vlines(x=arrival*1e6, ymin=y_min, ymax=y_max, linestyles="dashed", colors="black", alpha=0.75, label="Arrival Time = " + str(np.round((arrival*1e6), 3)) + r" $\mu$s")
            
            if (values['-XLOW-'] != '') & (values['-XHIGH-'] != ''):
                ax.set_xlim(float(values['-XLOW-']),float(values['-XHIGH-']))
            if (values['-YLOW-'] != '') & (values['-YHIGH-'] != ''):
                ax.set_ylim(float(values['-YLOW-']),float(values['-YHIGH-']))
            
            ax.set_xlabel(r"Time Shift ($\mu$s)")
            ax.set_ylabel("Normalized Amplitude")
            ax.set_title('Signal Overlap')
            ax.legend(loc="best")
            fig.tight_layout()
            canvas.draw()
    
    if event == '-SAVE-':
        file_path = sg.popup_get_file('Save as', no_window=True, save_as=True) + values['-SAVETYPE-']
        fig.savefig(file_path)
        
window.close()

#============================================================================#




