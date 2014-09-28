# Brandon Plaster & Alap Parikh
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
        #put_arr = np.genfromtxt(StringIO(raw_str),dtype=str,delimiter=",",autostrip=True, usecols=(5), skip_header=1) # Pick up time array
        dat_arr = np.genfromtxt(StringIO(raw_str),dtype=float,delimiter=",",autostrip=True, usecols=(8,9,10,11,12,13), skip_header=1) # Data array
        
    trip_disp = []  
    strt_time = []  
    outliers = []
    
    # Create Displacement Array & Find outliers
    for i, x in enumerate(dat_arr):
        try:
            disp = get_distance(x[3],x[2],x[5],x[4])  
            trip_disp = np.append(trip_disp,[disp],0)     
        except:
            outliers.append(i)
            
    # Remove Outliers from Arrays
    dat_arr = np.delete(dat_arr,(outliers),0)
    dat_arr = np.append(dat_arr, np.vstack(trip_disp), 1)
    strt_time = np.delete(strt_time,(outliers),0)
    
    # Percentile, mean and median
    #a = []
    #a = np.percentile(trip_disp, (1,99))
    #mean = np.mean(trip_disp)
    #std_dev = np.std(trip_disp)
    
    data = []
    data = dat_arr[:,6]
    
    #reject ouliers based on median
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    #data = data[s<30]
    #print len(data)
    #print max(data)
    
    # We assume here that people will not have the same pickup and dropoff location
    outliers = []
    for i,x in enumerate(s):
        if x > 6 or dat_arr[i,6] == 0:
            outliers.append(i)
    
    #Delete outliers
    dat_arr = np.delete(dat_arr,(outliers),0)
    
    #print "Total data points: ", len(trip_dist)
    #print "Maxiumum distance: ", max(trip_dist)," miles"
    #print "Minimum distance: ", min(trip_dist), " miles" 
    
        # Setup test and training data
    test_dat = dat_arr[::4,:2]
    train_dat = np.delete(dat_arr, np.arange(0,dat_arr.size,4),0)[:,:2]
    #print train_dat
    
    # Find weights for LS solution
    x = train_dat[:,1]
    y = train_dat[:,0]
    X = np.vstack([x,np.ones(x.size)]).T
    m, b = np.linalg.lstsq(X,y)[0] # Least squares fit
    # m1, b1 = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y) # How to get it based on slides
    # m2, b2 = np.dot(np.linalg.pinv(X),y) # How to get it with built in psuedo inverse function
    
    print "Training data LS weights: ", m, b
    #print m1, b1
    #print m2, b2
    
    # Plot LS fit for training data       
    plt.figure(4, figsize=(15,9))    
    plt.plot(x, y, 'o', label='Original data')
    plt.plot(x, m*x + b, 'r', label='Fitted line')
    plt.title('Trip Time vs. Trip Distance (Training Data)')
    plt.ylabel('Trip Time (in seconds)')
    plt.xlabel('Trip Distance (in miles)')
    plt.legend()
            
    #training_trip_time = train_dat[:,0]
    #training_trip_distance = train_dat[:,1]
    #print training_trip_time
    #print training_trip_distance
    
    #TRAINING_TRIP_TIME = sm.add_constant(training_trip_time)
    #model = sm.OLS(training_trip_distance, TRAINING_TRIP_TIME)
    #fit = model.fit()
    
    #print fit.summary()
    
    #plt.figure(5)
    #pylab.scatter(training_trip_time, training_trip_distance)
    #pylab.plot(training_trip_time, fit.fittedvalues)
    #pylab.show()
    
    # Error for Test
    x_test = test_dat[:,1]
    y_test = test_dat[:,0]
    y_fit_test = m*x_test+b
    OLS_error = np.sqrt(np.sum(np.square((y_fit_test-y_test)))/len(y_fit_test)) # OLS Error
    v = np.vstack([m,-1])
    r = np.vstack([x_test-x_test,y_fit_test-y_test])
    TLS_error = np.sqrt(np.sum(np.square(np.abs(np.dot(v.T,r[:,:]))/np.linalg.norm(v)))/len(y_fit_test)) # TLS Error
    print "OLS error (test data): ", OLS_error
    print "TLS error (test data): ", TLS_error
    
    #Crosscheck TLS using cosine theta
    OLS_offset = y_fit_test - y_test
    OLS_error_alt = np.sum(np.square(OLS_offset))
    cosine_theta = math.cos(math.atan(m))
    TLS_offset = cosine_theta * OLS_offset
    TLS_error_alt = np.sum(np.square(TLS_offset))
    print "OLS error (alternative way):", OLS_error_alt
    print "TLS error (alternative way):", TLS_error_alt
    #print OLS_error_alt*(cosine_theta**2)
    #print cosine_theta
    
    # Plot LS fit for test data       
    plt.figure(6, figsize=(15,9))    
    plt.plot(x_test, y_test, 'o', label='Original data')
    plt.plot(x_test, m*x_test + b, 'r', label='Fitted line')
    plt.title('Trip Time vs. Trip Distance (Test Data)')
    plt.ylabel('Trip Time (in seconds)')
    plt.xlabel('Trip Distance (in miles)')
    plt.legend()
    #plt.show()
    
        
if __name__ == '__main__':
  main()