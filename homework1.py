# Brandon Plaster
# CS 5785 Homework 1

import numpy as np
import math
import matplotlib.pyplot as plt
from distance import *
from StringIO import StringIO

def main():
    with open("example_data.csv") as f_in:
        raw_str = f_in.read()
        pu_arr_t = np.genfromtxt(StringIO(raw_str),dtype=None,delimiter=",",autostrip=True, usecols=(5), skip_header=1)
        dat_arr = np.genfromtxt(StringIO(raw_str),dtype=float,delimiter=",",autostrip=True, usecols=(8,9,10,11,12,13), skip_header=1)

        outliers = []
        distances = np.array([])  
        strt_time = np.array([])      
        
        for i, x in enumerate(dat_arr):
            try:
                distances = np.append(distances,[get_distance(x[3],x[2],x[5],x[4])],0)
            except:
                outliers.append(i)
                
        dat_arr = np.delete(dat_arr,(outliers),0)       
        #dat_arr = np.append(dat_arr,distances,1)
                
        print len(dat_arr[:,0]), len(distances)
        print distances
        trip_time = dat_arr[:,0]
        trip_dist = dat_arr[:,1]
        #trip_mag = dat_arr[:,6]
        
        #print len(trip_mag),max(trip_mag),min(trip_mag) # distance is measured in miles
        
       # Set up plots
        plt.figure(1, figsize=(15,8))
        plt.suptitle('All the things',weight='bold')
	
        sp1 = plt.subplot(311)
        #sp1.scatter(trip_time,pu_time)
        plt.xlabel('Trip Time (in seconds)')
        plt.ylabel('Pick Up Time')
        
        sp2 = plt.subplot(312)
        #sp2.scatter(trip_time,trip_dist)
        plt.xlabel('Trip Time (in seconds)')
        plt.ylabel('Trip Distance')
        
        sp3 = plt.subplot(313)
        #sp3.scatter(distances[:,1],distances[:,0])
        plt.xlabel('Trip Time (in seconds)')
        plt.ylabel('Distance from Pick up to Drop Off')
        
        plt.show()
        plt.savefig('homework1.png')
        
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