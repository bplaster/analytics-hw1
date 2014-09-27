# Brandon Plaster
# CS 5785 Homework 1

import pylab
import statsmodels.api as sm
import numpy as np
import math
import matplotlib.pyplot as plt
from distance import *
from StringIO import StringIO

def main():
    put_arr = []
    dat_arr = []
    with open("example_data.csv") as f_in:
        raw_str = f_in.read()
        put_arr = np.genfromtxt(StringIO(raw_str),dtype=str,delimiter=",",autostrip=True, usecols=(5), skip_header=1) # Pick up time array
        dat_arr = np.genfromtxt(StringIO(raw_str),dtype=float,delimiter=",",autostrip=True, usecols=(8,9,10,11,12,13), skip_header=1) # Data array
        
    trip_disp = []  
    strt_time = []  
    outliers = []
            
    # Create Start Time Array
    for x in put_arr:
        strt_time = np.append(strt_time,[get_time_as_float(x)],0)
    
    # Create Displacement Array & Find outliers
    for i, x in enumerate(dat_arr):
        try:
            disp = get_distance(x[3],x[2],x[5],x[4])
            # 1) Filter out when displacement is greater than distance
            # 2) Filter out when coordinates are 0
            if disp < x[1] and x[3] != 0: 
                trip_disp = np.append(trip_disp,[disp],0)
            else:
                outliers.append(i)        
        except:
            outliers.append(i)
        
    # Remove Outliers from Arrays
    dat_arr = np.delete(dat_arr,(outliers),0)
    strt_time = np.delete(strt_time,(outliers),0)
    
    # Arrays to plot
    trip_time = dat_arr[:,0]
    trip_dist = dat_arr[:,1]
    
    print len(trip_dist),max(trip_dist),min(trip_dist) # distance is measured in miles
    
    # Set up plots
    plt.figure(1, figsize=(15,9))
    plt.scatter(strt_time,trip_time)
    plt.xlabel('Pick Up Time (hour of day)')
    plt.ylabel('Trip Time (in seconds)')
    
    plt.figure(2, figsize=(15,9))
    plt.scatter(trip_time,trip_dist)
    plt.xlabel('Trip Time (in seconds)')
    plt.ylabel('Trip Distance (in miles)')
        
    plt.figure(3, figsize=(15,9))
    plt.scatter(trip_time,trip_disp)
    plt.xlabel('Trip Time (in seconds)')
    plt.ylabel('Trip Displacement (in miles)')
        
    #plt.show()
    
    # Setup test and training data
    test_dat = dat_arr[::4,:2]
    train_dat = np.delete(dat_arr, np.arange(0,dat_arr.size,4),0)[:,:2]
    #print train_dat
    
<<<<<<< Updated upstream
    # Find weights for LS solution
    x = train_dat[:,1]
    y = train_dat[:,0]
    X = np.vstack([x,np.ones(x.size)]).T
    m, b = np.linalg.lstsq(X,y)[0] # Least squares fit
    # m1, b1 = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y) # How to get it based on slides
    # m2, b2 = np.dot(np.linalg.pinv(X),y) # How to get it with built in psuedo inverse function
    
    print m, b
    #print m1, b1
    #print m2, b2
    
    # Plot LS fit        
    plt.figure(4, figsize=(15,9))    
    plt.plot(x, y, 'o', label='Original data')
    plt.plot(x, m*x + b, 'r', label='Fitted line')
    plt.ylabel('Trip Time (in seconds)')
    plt.xlabel('Trip Distance (in miles)')
    plt.legend()
    plt.show()
    
    
        
=======
    training_trip_time = train_dat[:,0]
    training_trip_distance = train_dat[:,1]
    #print training_trip_time
    #print training_trip_distance
    
    TRAINING_TRIP_TIME = sm.add_constant(training_trip_time)
    model = sm.OLS(training_trip_distance, TRAINING_TRIP_TIME)
    fit = model.fit()
    
    print fit.summary()
    
    plt.figure(4)
    pylab.scatter(training_trip_time, training_trip_distance)
    pylab.plot(training_trip_time, fit.fittedvalues)
    pylab.show()
>>>>>>> Stashed changes
        
def get_time_as_float (datetime):
    date, time = datetime.split(" ")
    hours, minutes, seconds = time.split(":")
    hours = float(hours)
    minutes = float(minutes)
    minutes = minutes/60
    hours = hours + minutes
    #hours = round(hours)
    return hours
    
        
if __name__ == '__main__':
  main()