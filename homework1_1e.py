# Brandon Plaster & Alap Parikh
# CS 5785 Homework 1e

#import pylab
#import statsmodels.api as sm
import numpy as np
import math
from distance import *
from StringIO import StringIO

def main():
    dat_arr = []
    with open("example_data.csv") as f_in:
        raw_str = f_in.read()
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
    
    # We assume here that people will not have the same pickup and dropoff location
    outliers = []
    for i,x in enumerate(s):
        if x > 6 or dat_arr[i,6] == 0:
            outliers.append(i)
    
    #Delete outliers
    dat_arr = np.delete(dat_arr,(outliers),0)
    
    # Setup test and training data
    test_dat = dat_arr[::4,:2]
    train_dat = np.delete(dat_arr, np.arange(0,dat_arr.size,4),0)[:,:2]
    #print train_dat
    
    # Find weights for LS solution
    x = train_dat[:,1]
    y = train_dat[:,0]
    X = np.vstack([x,np.ones(x.size)]).T
    m, b = np.linalg.lstsq(X,y)[0] # Least squares fit
    
    print "Training data LS weights: ", m, b
  
    #Error for training data
    y_fit_train = m*x + b
    OLS_error_train = np.sqrt(np.sum(np.square((y_fit_train - y)))/len(y_fit_train)) # OLS Error
    v_train = np.vstack([m,-1])
    r_train = np.vstack([x - x,y_fit_train-y])
    TLS_error_train = np.sqrt(np.sum(np.square(np.abs(np.dot(v_train.T,r_train[:,:]))/np.linalg.norm(v_train)))/len(y_fit_train)) # TLS Error
    print "OLS error (train data): ", OLS_error_train
    print "TLS error (train data): ", TLS_error_train
    
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
    OLS_error_alt = np.sqrt(np.sum(np.square(OLS_offset))/len(y_fit_test))
    cosine_theta = math.cos(math.atan(m))
    TLS_offset = cosine_theta * OLS_offset
    TLS_error_alt = np.sqrt(np.sum(np.square(TLS_offset))/len(y_fit_test))
    print "OLS error (alternative way):", OLS_error_alt
    print "TLS error (alternative way):", TLS_error_alt
    
        
if __name__ == '__main__':
  main()