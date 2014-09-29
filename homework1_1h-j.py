# Brandon Plaster & Alap Parikh
# CS 5785 Homework 1 1h-j

import numpy as np
from distance import *
from time import strftime, localtime

def main():
    print "Start time: ", strftime("%a, %d %b %Y %H:%M:%S",localtime()) 
    filename_train = "trip_data_1.csv"
    #filename_train = "example_data.csv"
    filename_test = "trip_data_2.csv"
    #filename_test = "example_data.csv"
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
    y_train = np.vstack(times)
    x_train = np.append(features,np.vstack(np.ones(features.shape[0])),1)
    weights = np.linalg.lstsq(x_train,y_train)[0] # Least squares fit
    print "OLS weights (training data): ", weights
    
    # Error for Training data
    p1x = np.vstack(np.zeros(7))
    p1y = np.sum(np.append(p1x,1)*weights.T)
    p1 = np.append(p1x, p1y)
    p2x = np.vstack(np.ones(7))
    p2y = np.sum(np.append(p2x,1)*weights.T)
    p2 = np.append(p2x, p2y)
    
    d_ols_sqrd_total = 0
    d_tls_sqrd_total = 0
        
    for i, x in enumerate(x_train):
        #OLS
        d_ols_sqrd = np.square(np.sum(x*weights.T,1)-y_train[i])
        d_ols_sqrd_total += d_ols_sqrd
        
        #TLS
        p = np.append(features[i],y_train[i],1)
        pa = p1 - p
        pb = p2 - p1
        pbn = np.square(np.linalg.norm(pb))
        d_tls_sqrd = ((np.square(np.linalg.norm(pa))*pbn) - np.square(np.dot(pa,pb)))/pbn
        d_tls_sqrd_total += d_tls_sqrd
    
    OLS_error = np.sqrt(d_ols_sqrd_total/y_train.shape[0]) # OLS Error
    TLS_error = np.sqrt(d_tls_sqrd_total/y_train.shape[0]) # TLS Error
    
    print "OLS error (training data): ", OLS_error[0]
    print "TLS error (training data): ", TLS_error

    # Test Set  
    test_x_sum = np.zeros(8)
    d_ols_sqrd_total_test = 0
    d_tls_sqrd_total_test = 0 
    count_test = 0
    with open(filename_test) as f_in:
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
                time_test = float(line[8])
                disp = get_distance(pu_lat,pu_long,do_lat,do_long)
                # 1) Filter out when displacement is greater than distance
                # 2) Filter out when coordinates are 0
                if disp < dist and pu_long != 0: 
                    feat_vect = np.array((dist, disp, pu_lat, pu_long, do_lat, do_long, pu_time))
                    feat_vect = feat_scale(feat_vect,min_feat,range_feat)
                    
                    #OLS
                    x_test = np.append(feat_vect,np.ones(1),1)
                    d_ols_sqrd = np.square(np.sum(x_test*weights.T,1)-time_test)
                    d_ols_sqrd_total_test += d_ols_sqrd

                    #TLS
                    p = np.append(feat_vect,np.array(time_test))
                    p = np.array(p)
                    pa = p1 - p
                    pb = p2 - p1
                    pbn = np.square(np.linalg.norm(pb))
                    d_tls_sqrd = ((np.square(np.linalg.norm(pa))*pbn) - np.square(np.dot(pa,pb)))/pbn
                    d_tls_sqrd_total_test += d_tls_sqrd
                    
                    test_x_sum += p
                    count_test += 1
            except:
                pass

    OLS_error_test = np.sqrt(d_ols_sqrd_total/count_test) # OLS Error
    TLS_error_test = np.sqrt(d_tls_sqrd_total/count_test) # TLS Error
    
    test_x_mean = np.vstack(test_x_sum/count_test)
        
    print "OLS error (test data): ", OLS_error_test[0]
    print "TLS error (test data): ", TLS_error_test
    
    coef_sum = np.zeros([8,8])
    diff_x_sum = np.zeros([8,1])
    
    # Correlation Coefficiant test data
    with open(filename_test) as f_in:
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
                time_test = float(line[8])
                disp = get_distance(pu_lat,pu_long,do_lat,do_long)
                # 1) Filter out when displacement is greater than distance
                # 2) Filter out when coordinates are 0
                if disp < dist and pu_long != 0: 
                    feat_vect = np.array((dist, disp, pu_lat, pu_long, do_lat, do_long, pu_time))
                    feat_vect = feat_scale(feat_vect,min_feat,range_feat)
                    p = np.vstack(np.append(feat_vect,np.array(time_test)))
                    x_diff_test = p-test_x_mean
                    coef_sum += x_diff_test*x_diff_test.T
                    diff_x_sum += np.square(x_diff_test)
            except:
                pass
    
    # Correlation Coefficiant test data
    coef = (coef_sum)/(np.sqrt(diff_x_sum)*np.sqrt(diff_x_sum).T)
    print "Training Coefficient (test data): ", coef

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