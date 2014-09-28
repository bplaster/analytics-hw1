# Brandon Plaster & Alap Parikh
# CS 5785 Homework 1

import numpy as np
from distance import *
from StringIO import StringIO
from time import strftime, localtime

def main():
    print "Start time: ", strftime("%a, %d %b %Y %H:%M:%S",localtime())
    dat_arr = []
    trip_disp = []  
    outliers = []
    with open("example_data.csv") as f_in:
        dat_arr = np.genfromtxt(StringIO(f_in.read()),dtype=float,delimiter=",",autostrip=True, usecols=(8,9,10,11,12,13), skip_header=1) # Data array
    
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
    
    # Find weights for LS solution
    x = dat_arr[:,1]
    y = dat_arr[:,0]
    X = np.vstack([x,np.ones(x.size)]).T
    m, b = np.linalg.lstsq(X,y)[0] # Least squares fit
    
    print "OLS weights (training data): ", m, b
    
    # Error for Training data
    y_fit_train = m*x+b
    OLS_error = np.sqrt(np.sum(np.square((y_fit_train-y)))/len(y_fit_train)) # OLS Error
    v = np.vstack([m,-1])
    r = np.vstack([np.zeros(x.size),y_fit_train-y])
    TLS_error = np.sqrt(np.sum(np.square(np.abs(np.dot(v.T,r[:,:]))/np.linalg.norm(v)))/len(y_fit_train)) # TLS Error
    print "OLS error (training data): ", OLS_error
    print "TLS error (training data): ", TLS_error
    
    # Correlation Coefficiant training data
    x_diff = x-np.mean(x)
    y_diff = y-np.mean(y)
    coef = (np.sum(x_diff*y_diff))/(np.sqrt(np.sum(np.square(x_diff)))*np.sqrt(np.sum(np.square(y_diff))))
    print "Training Coefficient (training data): ", coef
    
    # Read in test data
    OLS_sum = 0
    TLS_sum = 0
    test_length = 0
    test_x_sum = 0
    test_y_sum = 0
    coef_sum = 0
    diff_x_sum = 0
    diff_y_sum = 0
    filename = "trip_data_2.csv"

    # Error for Test data
    with open(filename) as f_in:
        next(f_in)
        for line in f_in:
            line = line.strip().split(',')
            y_test = float(line[8])
            x_test = float(line[9])
            y_fit_test = m*x_test+b
            OLS_sum += np.square(y_fit_test-y_test)
            r_test = np.vstack([0,y_fit_test-y_test])
            TLS_sum += np.square(np.abs(np.dot(v.T,r_test))/np.linalg.norm(v))
            test_length += 1
            test_x_sum += x_test
            test_y_sum += y_test

    OLS_error = np.sqrt(OLS_sum/test_length) # OLS Error
    TLS_error = np.sqrt(TLS_sum/test_length) # TLS Error
    print "OLS error (test data): ", OLS_error
    print "TLS error (test data): ", TLS_error[0][0]
    test_x_mean = test_x_sum/test_length
    test_y_mean = test_y_sum/test_length
            
    # Correlation Coefficiant test data
    with open(filename) as f_in:
        next(f_in)
        for line in f_in:
            line = line.strip().split(',')
            y_test = float(line[8])
            x_test = float(line[9])
            x_diff_test = x_test-test_x_mean
            y_diff_test = y_test-test_y_mean
            coef_sum += x_diff_test*y_diff_test
            diff_x_sum += np.square(x_diff_test)
            diff_y_sum += np.square(y_diff_test)
    
    # Correlation Coefficiant test data
    coef = (coef_sum)/(np.sqrt(diff_x_sum)*np.sqrt(diff_y_sum))
    print "Training Coefficient (test data): ", coef
    print "End time: ", strftime("%a, %d %b %Y %H:%M:%S",localtime())
        
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