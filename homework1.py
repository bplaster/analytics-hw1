# Brandon Plaster
# CS 5785 Homework 1

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