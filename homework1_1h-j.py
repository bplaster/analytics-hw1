# Brandon Plaster & Alap Parikh
# CS 5785 Homework 1

import numpy as np
from distance import *
from time import strftime, localtime

def main():
    print "Start time: ", strftime("%a, %d %b %Y %H:%M:%S",localtime()) 
    filename_train = "trip_data_1.csv"
    filename_train = "example_data.csv"
    filename_test = "example_data.csv"
    max_feat = -9999 * np.ones(7)
    min_feat = 9999 * np.ones(7)
    features = []
    times = []
        
    # Get the Range of each feature    
    with open(filename_train) as f_in:
        next(f_in)
        for line in f_in:
            try:
                line = line.strip().split(',')
                pu_time = get_time_as_float(line[5])
                dist = float(line[9])
                pu_long = float(line[10])
                pu_lat = float(line[11])
                do_long = float(line[12])
                do_lat = float(line[13])
                disp = get_distance(pu_lat,pu_long,do_lat,do_long)
                # 1) Filter out when displacement is greater than distance
                # 2) Filter out when coordinates are 0
                if disp < dist and pu_long != 0: 
                    feat_vect = np.array((dist, disp, pu_lat, pu_long, do_lat, do_long, pu_time))
                    for i, x in enumerate(feat_vect):
                        max_feat[i] = max(max_feat[i],x)
                        min_feat[i] = min(min_feat[i],x)
            except:
                pass
    
    print "Max features: " , max_feat
    print "Min features: " , min_feat

    range_feat = max_feat - min_feat

    # Train Features    
    with open(filename_train) as f_in:
        next(f_in)
        for line in f_in:
            try:
                line = line.strip().split(',')
                pu_time = get_time_as_float(line[5])
                dist = float(line[9])
                pu_long = float(line[10])
                pu_lat = float(line[11])
                do_long = float(line[12])
                do_lat = float(line[13])
                time_train = float(line[8])
                disp = get_distance(pu_lat,pu_long,do_lat,do_long)
                # 1) Filter out when displacement is greater than distance
                # 2) Filter out when coordinates are 0
                if disp < dist and pu_long != 0: 
                    feat_vect = np.array((dist, disp, pu_lat, pu_long, do_lat, do_long, pu_time))
                    feat_vect = feat_scale(feat_vect,min_feat,range_feat)
                    features.append(feat_vect)
                    times.append(time_train)
            except:
                pass

    # Find weights for LS solution
    features = np.array(features)
    y = np.vstack(times)
    x = np.append(features,np.vstack(np.ones(features.shape[0])),1)
    weights = np.linalg.lstsq(x,y)[0] # Least squares fit
    print "OLS weights (training data): ", weights
    
    # Error for Training data
    y_fit_train = np.vstack(np.sum(x*weights.T,1))
    print y_fit_train.shape, y.shape
    OLS_error = np.sqrt(np.sum(np.square((y_fit_train-y)))/y_fit_train.shape[0]) # OLS Error
    print "OLS error (training data): ", OLS_error
    #print "TLS error (training data): ", TLS_error
    
    print "End time: ", strftime("%a, %d %b %Y %H:%M:%S",localtime())
    
def feat_scale (x, x_min, x_range):
    return (x-x_min)/x_range

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