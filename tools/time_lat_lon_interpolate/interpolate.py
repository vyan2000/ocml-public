#import sys
#print(sys.version)

import pandas as pd
import xarray as xr
import numpy as np

from matplotlib import pyplot as plt

#import os
#print("current working directory is", os.getcwd())


##########################
##########################
##########################

def sel_points_multilinear_dist_lat_lon(dframe_in, dims='points', col_name = 'str'):
    # sel_points_multilinear_lat_lon(distance_to_coast, dframe, dims = 'points', col_name ='dist')
    # col_name should match the satellite dataset
    '''
    # this is a file for interpolating the distance to land
    * given (lat, lon) pair, we can identify its closest distance to land
    * the file is located at original 0.04-degree data set:
    * Docs: http://oceancolor.gsfc.nasa.gov/cms/DOCS/DistFromCoast
    * Sources: http://oceancolor.gsfc.nasa.gov/DOCS/DistFromCoast/dist2coast.txt.bz2
    * It is done by using the 'distance to coast' dataset
    * based on function for generating multilinear interpolations in python
    
    input:  
    dframe    pandas.dataframe; the lists {lat, lon}
    dims      default to be "Points" of xarray.dataset
    col_name  the variable to interpolate;
    
    output: dframe_out {id, time, lat, lon, distance} 
    '''

    '''
    # 2D
    wx = delta_x / (x_next - x_nearest)
    wy = delta_y / (y_next - y_nearest)
    f = (1-wx)*(1-wy)*da.isel(x=ix_nearest, y=iy_nearest).values +
            (1-wx)*wy*da.isel(x=ix_nearest,y=iy_next).values     +
           wx*(1-wy)*da.isel(x=ix_next,y=iy_nearest).values     +
               wx*wy*da.isel(x=ix_next,y=iy_next).values
    '''

    # the resolution is 0.01 degree, which is 1 km
    dist_db = pd.read_csv("./dist2land_data/dist2coast.txt.bz2", header=None,
                          sep='\t', names=['lon','lat','dist'])
    
    # transfer lon from the range [-180, 180] to [0,360]
    mask= dist_db.lon<0
    dist_db.lon[mask] = dist_db.loc[mask].lon + 360
    print('after processing, the minimum longitude is %f4.3 and maximum is %f4.3' 
          % (dist_db.lon.min(),dist_db.lon.max()) )
   
    # Select only the arabian sea region
    arabian_sea = (dist_db.lon > 45) & (dist_db.lon< 75) & (dist_db.lat> 5) & (dist_db.lat <28)
    dist_db_arabian = dist_db[arabian_sea]
    print('dist_db.shape is %s, dist_db_arabian.shape is %s' % (dist_db.shape, dist_db_arabian.shape) )
   
    # visualize the unsigned(in-land & out-land) distance around global region
    fig, axis = plt.subplots(figsize=(12,8))
    dist_db_arabian.plot(kind='scatter', x='lon', y='lat', c='dist', cmap='RdBu_r', 
                         edgecolor='none', ax=axis, title='distance to the nearest coast',
                         fontsize=12)
    #plt.savefig('distance_to_coast_main.png')  # for plot, to be removed!!
    plt.show()
    plt.close()

    # transfer the dataframe into dataset, and to prepare for interpolation
    # set time & id as the index); use reset_index to revert
    dist_DS = xr.Dataset.from_dataframe(dist_db_arabian.set_index(['lon','lat']) )
    # lat -- asceding
    # lon -- ascending

    print("\n ******** Interpolation of distance using (lat, lon) ******* \n")
    ###
    eps_float32 = np.finfo(np.float32).eps   # selection on the nearest point will reduce accuracy!

    ## get the indices of time, lat, lon
    idx_lat = dist_DS.indexes['lat']
    idx_lon = dist_DS.indexes['lon']

    ## bounds
    '''
    # row_case.lat > dist_DS.lat.min   # ascending
    # row_case.lat < dist_DS.lat.max   # ascending
    # row_case.lon > ddist_DS.lat.min  # ascending
    # row_case.lat > dist_DS.lat.min   # ascending
    '''
    dset_latmin = dist_DS.lat.to_series().min()
    dset_latmax = dist_DS.lat.to_series().max()

    dset_lonmin = dist_DS.lon.to_series().min()
    dset_lonmax = dist_DS.lon.to_series().max()

    mask = (dframe_in.lat > dset_latmin) & (dframe_in.lat < dset_latmax) # [True, True, True]
    mask = mask & (dframe_in.lon > dset_lonmin) & (dframe_in.lon < dset_lonmax)

    dframe = dframe_in.loc[mask,:]

    ################### printings
    ### 'lat is ascending
    #print(dist_DS.indexes['lat'])
    #print(type(dist_DS.indexes['lat']))

    ### 'lon is ascending
    #print(dist_DS.indexes['lon'])
    #print(type(dist_DS.indexes['lon']))

    # validation: to locate the 'dist' for the test case 2

    ####
    # interpolation on the lat dimension
    lat_len = len(dframe.lat.values)

    '''caution: cannot do this inside the function get_loc,
    see https://github.com/pandas-dev/pandas/issues/3488
    '''
    # xlat_test = dframe.lat.values + 0.06
    # base [ 5.20833349  5.29166174]
    # cell distance around .8, use .2 & .6 as two tests
    xlat_test = dframe.lat.values
    #print('\n xlat_test \n', xlat_test)
    # xlat_test [ 5.03  5.07  5.11]

    ilat_nearest = [idx_lat.get_loc(xlat_test[i], method='nearest')
                    for i in range(0, lat_len)]
    #print('\n ilat_nearest \n', ilat_nearest) # [0, 1, 2]

    xlat_nearest = dist_DS.lat[ilat_nearest].values
    #print('\n xlat_nearest \n', xlat_nearest) #    [ 5.02  5.06  5.1 ]

    delta_xlat = xlat_test - xlat_nearest
    #print("\n delta_xlat \n",delta_xlat)      #  [ 0.01  0.01  0.01]


    # the nearest index is on the left; it is ascending
    # delta_xlat[i] is of type float64
    ilat_next = [ilat_nearest[i] + 1 if delta_xlat[i] >= (1.0 * eps_float32)
                 else ilat_nearest[i] - 1
                 for i in range(0, lat_len)]
    #print('\n ilat_next \n', ilat_next)  # [1, 2, 3]

    # find the next coordinates value
    xlat_next = dist_DS.lat[ilat_next].values
    #print('\n xlat_next \n', xlat_next)  #  [ 5.06  5.1   5.14]

    # prepare for the Tri-linear interpolation
    w_lat = delta_xlat / (xlat_next - xlat_nearest)
    #print('\n w_lat \n', w_lat) #  [ 0.25  0.25  0.25]

    ####
    # interpolation on the lon dimension
    # xlon_test = dframe.lon.values +0.03
    # base [74.7083358765, 74.6250076294]
    # cell distance around .4, use .1 & .3 as two tests
    xlon_test = dframe.lon.values
    #print('\n xlon_test \n', xlon_test)  #  [ 45.05  45.09  45.13]

    ilon_nearest = [idx_lon.get_loc(xlon_test[i], method='nearest')
                    for i in range(0, lat_len)]
    #print('\n ilon_nearest \n', ilon_nearest) #  [1, 2, 3]

    xlon_nearest = dist_DS.lon[ilon_nearest].values
    #print('\n xlon_nearest \n', xlon_nearest) #  [ 45.06  45.1   45.14]

    delta_xlon = xlon_test - xlon_nearest
    #print("\n delta_xlon \n", delta_xlon)     #   [-0.01 -0.01 -0.01]

    ilon_next = [ilon_nearest[i] + 1 if delta_xlon[i] >= (1.0 * eps_float32)
                 else ilon_nearest[i] - 1
                 for i in range(0, lat_len)]
    #print('\n ilon_next \n',ilon_next)  # [0, 1, 2]

    # find the next coordinate values
    xlon_next = dist_DS.lon[ilon_next].values
    #print("\n xlon_next \n", xlon_next) #  [ 45.02  45.06  45.1 ]

    # prepare for the Tri-linear interpolation
    w_lon = delta_xlon / (xlon_next - xlon_nearest)
    #print("\n w_lon \n", w_lon) #  [ 0.25  0.25  0.25]

    ####
    # local Tensor product for Trilinear interpolation
    # caution: nan values, store as "list_of_array to 2d_array" first, then sum

    # no casting to list needed here, inputs are already lists
    tmp = np.array([
        dist_DS[col_name].isel_points(lat=ilat_nearest, lon=ilon_nearest).values,
        dist_DS[col_name].isel_points(lat=ilat_nearest, lon=ilon_next).values,
        dist_DS[col_name].isel_points(lat=ilat_next, lon=ilon_nearest).values,
        dist_DS[col_name].isel_points(lat=ilat_next, lon=ilon_next).values])

    weights = np.array([ (1 - w_lat) * (1 - w_lon),
                         (1 - w_lat) * w_lon,
                         w_lat * (1 - w_lon),
                         w_lat * w_lon  ]
                       )

    # how to deal with "nan" values, fill in missing values for the np.array tmpAll
    # or fill the mean values to the unweighted array
    # http://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns

    #print('\n neighbouring tensor used \n', tmp)
    '''
     neighbouring tensor used
     [[ 277.093  276.822  276.561]
      [ 280.401  280.076  279.822]
      [ 280.076  279.822  279.561]
      [ 283.349  283.081  282.821]]
    '''

    # column min: (nan+0.245878 + nan + nan + 0.19680101 + nan +  nan + 0.18532801)/8 = 0.20933567333
    col_mean = np.nanmean(tmp, axis=0)
    #print('\n its mean along axis 0(column) \n', col_mean)  #  [ 280.22975  279.95025  279.69125]


    # filling the missing values.
    inds = np.where(np.isnan(tmp))
    #print('\n nan index\n', inds)
    tmp[inds] = np.take(col_mean, inds[1])
    #print('\n values after the fill \n', tmp)

    #print('\n weighting tensor used \n', weights)

    #print("weights.shape", weights.shape) #  (4, 3)
    #print("tmp.shape", tmp.shape)  #  (4, 3)

    nrow_w, ncol_w = weights.shape
    nrow_t, ncol_t = tmp.shape
    assert nrow_w == nrow_t, "the row count of weights and values are not the same!"
    assert ncol_w == ncol_t, "the row count of weights and values are not the same!"
    #print('\n tensor product\n', np.dot(weights[:,0], tmp[:,0]) ) #  278.6635625 should be [  278.6635625]

    # new interpolation process of the Chl_a
    dist_new = np.empty(ncol_w)
    for i in range(0, ncol_w, 1):
        dist_new[i] = np.dot(weights[:, i], tmp[:, i])

    #print('dist_new_Int',  dist_new) #  [[ 278.6635625  278.3858125  278.1261875]
    # validate by 1D array
    # val = np.array([277.093,280.401, 280.076, 283.349])
    # np.dot(val, weights) # 278.6635625

    ###--------- approach depreciated ------------###
    ###--------- use df to generalize ------------###
    ## output xarray.dataarray of points, see examples below
    # this is the way how xarray.Dataset works
    # if you want a xarray.DataArray, first generate a xarray.Dataset, then select DataArray from there.
    #dframe_out = xr.Dataset({col_name: (['points'], dist_new)},
    #                        coords={
    #                            # 'time':(['points'],['2002-07-13 00:00:00', '2002-07-22 00:00:00', '2002-07-13 00:00:00']) ,
    #                            'time': (['points'], dframe.time),
    #                            'id': (['points'], dframe.id),
    #                            'lon': (['points'], dframe.lon),
    #                            'lat': (['points'], dframe.lat),
    #                            'points': (['points'], range(0, len(dframe)))})  # dims is set to point
    ###-------------------------------------------### 
    #http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    dframe.loc[:, col_name] = pd.Series(dist_new, index = dframe.index)
    #print('\n', dframe[col_name])
    return dframe








'''
# test case
# input: {lat, lon, time, id}
# output: dataframe with {distance, lat, lon, time, id} ready for merge
## test case 2: take 3 entries (passed):
row_case2 =  pd.DataFrame(data = {'time':['2002-07-13 00:00:00', '2002-07-22 00:00:00', '2002-07-13 00:00:00'] ,
                          'id': [10206, 10206, 10206], 'lon':[45.02, 45.06, 45.1],
                          'lat':[5.02, 5.06, 5.1]}, index=[1,2,3])

row_case2['lat'] =   row_case2['lat'] + 0.01
row_case2['lon'] =   row_case2['lon'] + 0.03

result_out2 = sel_points_multilinear_dist_lat_lon(row_case2, dims='points', col_name = 'dist')
# result_out3 should have {lat, lon, time, id, distance}
'''









def sel_points_multilinear_time_lat_lon(dset, dframe_in, dims='points', col_name = 'str'): 
    # col_name should match the satellite dataset
    '''    
    sel_points_multilinear_time_lat_lon(ds_9day, row_case1, dims = 'points', col_name ='chlor_a')
    
    function for generating multilinear interpolations in python
    
    input:  dset      xarray.dataset; the dictionary of values 
            dframe_in    pandas.dataframe; the lists {time, lat, lon, id}
            dims      default to be "Points" of xarray.dataset
            col_name  the variable to interpolate;
       
    output: dframe 
    '''


    '''
    ### code piece for Trilinear interpolation
    # 1D
    w = delta_x / (x_next -x_nearest)
    f = (1-w)*da.isel(x=ix_nearest).values + w* da.isel(x=ix_next).values

    # 2D
    wx = delta_x / (x_next - x_nearest)
    wy = delta_y / (y_next - y_nearest)
    f = (1-wx)*(1-wy)*da.isel(x=ix_nearest, y=iy_nearest).values +
            (1-wx)*wy*da.isel(x=ix_nearest,y=iy_next).values     +
           wx*(1-wy)*da.isel(x=ix_next,y=iy_nearest).values     +
               wx*wy*da.isel(x=ix_next,y=iy_next).values

    # 3D
    ### interpolate the Trilinear Value
    dimensions 3 : time, lat, lon
    locations: xtime_test, xlat_test, xlon_test
    weights(from the nearest coordinates): w_time, w_lat, w_lon
    output: a singel value based on 1D interpolation on each dimension, so use 8-neighbours

    wx = delta_x / (x_next - x_nearest)
    wy = delta_y / (y_next - y_nearest)
    wz = delta_z / (z_next - z_nearest)
    f = (1-wx)*(1-wy)*(1-wz)*da.isel(x=ix_nearest, y=iy_nearest, z=iz_nearest).values  +
            (1-wx)*(1-wy)*wz*da.isel(x=ix_nearest, y=iy_nearest, z=iz_next).values     +
            (1-wx)*wy*(1-wz)*da.isel(x=ix_nearest, y=iy_next, z=iz_nearest).values     +
                (1-wx)*wy*wz*da.isel(x=ix_nearest, y=iy_next, z=iz_next).values        +
            wx*(1-wy)*(1-wz)*da.isel(x=ix_next, y=iy_nearest, z=iz_nearest).values     +
                wx*(1-wy)*wz*da.isel(x=ix_next, y=iy_nearest, z=iz_next).values        +
                wx*wy*(1-wz)*da.isel(x=ix_next, y=iy_next, z=iz_nearest).values        +
                    wx*wy*wz*da.isel(x=ix_next, y=iy_next, z=iz_next).values
    '''

    ###
    eps_float32 = np.finfo(np.float32).eps   # selection on the nearest point will reduce accuracy!

    ## get the indices of time, lat, lon
    idx_time = dset.indexes['time'] 
    idx_lat = dset.indexes['lat']
    idx_lon = dset.indexes['lon']



    ##  bounds
    '''
    # row_case.lat > ds_9day.lat.min  # descending
    # row_case.lat < ds_9day.lat.max  # descending
    # row_case.lon > ds_9day.lat.min  # ascending
    # row_case.lat > ds_9day.lat.min  # ascending
    '''
    dset_latmin = dset.lat.to_series().min()
    dset_latmax = dset.lat.to_series().max()

    dset_lonmin = dset.lon.to_series().min()
    dset_lonmax = dset.lon.to_series().max()

    mask = (dframe_in.lat > dset_latmin) & (dframe_in.lat < dset_latmax)
    mask = mask & (dframe_in.lon > dset_lonmin) & (dframe_in.lon < dset_lonmax)

    dframe= dframe_in.loc[mask,:]
    
    '''
    ################### printings
    ### 'time is from small to big number'
    print(dset.indexes['time'])
    print(type(dset.indexes['time']))

    ### 'lat is from *big* to small number'
    print(dset.indexes['lat'])
    print(type(dset.indexes['lat']))

    ### 'lon is from small to big number'
    print(dset.indexes['lon'])
    print(type(dset.indexes['lon']))

    # validation: to locate the chl-a for this point 
    #{time:2002-07-13, lat:27.7916660309, lon:45.2916679382}
    '''
    
    #### 
    #interpolation on the time dimension
    time_len = len(dframe.time.values)
    xtime_test = list([ np.datetime64(dframe.time.values[i]) 
                 for i in range(0,time_len)  ] )  # for delta 
    #print('\n xtime_test \n', xtime_test)

    '''caution: cannot do this inside the function get_loc,
    see https://github.com/pandas-dev/pandas/issues/3488
    '''
    itime_nearest = [idx_time.get_loc(xtime_test[i], method='nearest') 
                     for i in range(0, time_len)]
    #print('\n itime_nearest \n', itime_nearest)  # [1,2]

    xtime_nearest =  dset.time[itime_nearest].values  
    #  ['2002-07-13T00:00:00.000000000' '2002-07-22T00:00:00.000000000']
    #print('\n xtime_nearest\n', xtime_nearest)
    # ['2002-07-13T00:00:00.000000000' '2002-07-22T00:00:00.000000000']
    #print('xtime_nearest', type(xtime_nearest))
    # xtime_nearest <class 'numpy.ndarray'> # time_nearest <class 'numpy.datetime64'>

    # the time distance in days
    delta_xtime = (xtime_test - xtime_nearest) / np.timedelta64(1, 'D')
    #print('\n delta_xtime in days \n', delta_xtime)
    #print(type(delta_xtime))

    itime_next = [itime_nearest[i]+1 if delta_xtime[i] >=0
                                     else itime_nearest[i]-1
                                     for i in range(0, time_len) ]
    #print('\n itime_next \n',itime_next)  # [2, 3]

    # find the next coordinate values
    xtime_next = dset.time[itime_next].values
    #print('\n xtime_next \n', xtime_next)
    # ['2002-07-22T00:00:00.000000000' '2002-07-31T00:00:00.000000000']

    # prepare for the Tri-linear interpolation
    base_time = (xtime_next - xtime_nearest) / np.timedelta64(1, 'D')  
    # [ 9.  9.]
    #print('\n base_time \n', base_time)
    w_time = delta_xtime / base_time  
    #print('\n w_time \n', w_time) # [ 0.  0.]


    #### 
    #interpolation on the lat dimension
    #xlat_test = dframe.lat.values + 0.06  
    # base [ 5.20833349  5.29166174] 
    # cell distance around .8, use .2 & .6 as two tests
    xlat_test = dframe.lat.values
    #print('\n xlat_test \n', xlat_test)
    # xlat_test [ 5.26833349  5.35166174]
    
    ilat_nearest = [idx_lat.get_loc(xlat_test[i], method='nearest') 
                    for i in range(0, time_len)]
    #print('\n ilat_nearest \n', ilat_nearest) # [272, 271]

    xlat_nearest = dset.lat[ilat_nearest].values  
    #print('\n xlat_nearest \n', xlat_nearest) # [ 5.29166174  5.37499762]

    delta_xlat = xlat_test - xlat_nearest
    #print("\n delta_xlat \n",delta_xlat)      #  [-0.02332825 -0.02333588]


    # the nearest index is on the right; but order of the latitude is different, it is descending
    # delta_xlat[i] is of type float64
    ilat_next = [ilat_nearest[i]-1 if delta_xlat[i] >= (-1.0 *eps_float32)
                 else ilat_nearest[i]+1  
                 for i in range(0, time_len) ]
    #print('\n ilat_next \n', ilat_next)  # [273, 272]

    # find the next coordinates value
    xlat_next = dset.lat[ilat_next].values
    #print('\n xlat_next \n', xlat_next)  # [ 5.20833349  5.29166174]

    # prepare for the Tri-linear interpolation
    w_lat = delta_xlat / (xlat_next - xlat_nearest)
    #print('\n w_lat \n', w_lat) # [ 0.27995605  0.28002197]

    #### 
    #interpolation on the lon dimension
    #xlon_test = dframe.lon.values +0.06
    # base [74.7083358765, 74.6250076294] 
    # cell distance around .8, use .2 & .6 as two tests
    xlon_test = dframe.lon.values
    #print('\n xlon_test \n', xlon_test)  # [ 74.76833588  74.68500763]

    ilon_nearest = [idx_lon.get_loc(xlon_test[i], method='nearest') 
                    for i in range(0, time_len)]
    #print('\n ilon_nearest \n', ilon_nearest) # [357, 356]

    xlon_nearest = dset.lon[ilon_nearest].values  
    #print('\n xlon_nearest \n', xlon_nearest) # [ 74.79166412  74.70833588]

    delta_xlon = xlon_test - xlon_nearest     
    #print("\n delta_xlon \n", delta_xlon)     #  [-0.02332825 -0.02332825]

    ilon_next = [ilon_nearest[i]+1 if delta_xlon[i] >= (1.0 *eps_float32)
                 else ilon_nearest[i]-1  
                 for i in range(0, time_len) ]
    #print('\n ilon_next \n',ilon_next)  # [356, 355]

    # find the next coordinate values
    xlon_next = dset.lon[ilon_next].values
    #print("\n xlon_next \n", xlon_next) # [ 74.70833588  74.62500763]

    # prepare for the Tri-linear interpolation
    w_lon = delta_xlon / (xlon_next - xlon_nearest)
    #print("\n w_lon \n", w_lon) # [ 0.27995605  0.27995605]

    ####
    # local Tensor product for Trilinear interpolation
    # caution: nan values, store as "list_of_array to 2d_array" first, then sum

    # no casting to list needed here, inputs are already lists
    tmp = np.array([
             dset[col_name].isel_points(time=itime_nearest, lat=ilat_nearest, lon=ilon_nearest).values,
             dset[col_name].isel_points(time=itime_nearest, lat=ilat_nearest, lon=ilon_next).values,
             dset[col_name].isel_points(time=itime_nearest, lat=ilat_next, lon=ilon_nearest).values,
             dset[col_name].isel_points(time=itime_nearest, lat=ilat_next, lon=ilon_next).values,
             dset[col_name].isel_points(time=itime_next, lat=ilat_nearest, lon=ilon_nearest).values,
             dset[col_name].isel_points(time=itime_next, lat=ilat_nearest, lon=ilon_next).values,
             dset[col_name].isel_points(time=itime_next, lat=ilat_next, lon=ilon_nearest).values,
             dset[col_name].isel_points(time=itime_next, lat=ilat_next, lon=ilon_next).values ])

    weights = np.array([(1-w_time)*(1-w_lat)*(1-w_lon),
                        (1-w_time)*(1-w_lat)*w_lon,
                        (1-w_time)*w_lat*(1-w_lon),
                        (1-w_time)*w_lat*w_lon,
                         w_time*(1-w_lat)*(1-w_lon),
                         w_time*(1-w_lat)*w_lon,
                         w_time*w_lat*(1-w_lon),
                         w_time*w_lat*w_lon ])


    # how to deal with "nan" values, fill in missing values for the np.array tmpAll 
    # or fill the mean values to the unweighted array
    # http://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns

    #print('\n neighbouring tensor used \n', tmp)
    '''
     neighbouring tensor used 
     [[        nan  0.181841  ]
     [ 0.245878           nan]
     [        nan         nan]
     [        nan         nan]
     [ 0.19680101         nan]
     [        nan         nan]
     [        nan         nan]
     [ 0.18532801         nan]]
    '''

    # column min: (nan+0.245878 + nan + nan + 0.19680101 + nan +  nan + 0.18532801)/8 = 0.20933567333
    col_mean = np.nanmean(tmp, axis=0)
    #print('\n its mean along axis 0(column) \n', col_mean)  #  [ 0.20933567  0.181841  ]


    # filling the missing values.
    inds = np.where(np.isnan(tmp))
    #print('\n nan index\n', inds)
    tmp[inds]=np.take(col_mean, inds[1])
    #print('\n values after the fill \n', tmp)

    #print('\n weighting tensor used \n', weights)

    #print("weights.shape", weights.shape) # (8, 3)
    #print("tmp.shape", tmp.shape)  # (8, 3)

    
    nrow_w, ncol_w = weights.shape
    nrow_t, ncol_t = tmp.shape
    assert nrow_w == nrow_t, "the row count of weights and values are not the same!"
    assert ncol_w == ncol_t, "the row count of weights and values are not the same!"
    #print('\n tensor product\n', np.dot(weights[:,0], tmp[:,0]) ) # 0.216701896135 should be [ 0.2167019]

    # new interpolation process of the Chl_a
    var_new = np.empty(ncol_w)
    for i in range(0, ncol_w, 1):
        var_new[i] =  np.dot(weights[:,i], tmp[:,i])

    #print('chl_newInt',  chl_new) #  [ 0.2167019  0.181841   0.2167019]
    # validate by 1D array
    # val = np.array([0.20933567, 0.245878,  0.20933567,
    #                0.20933567, 0.19680101, 0.20933567,
    #               0.20933567,0.18532801]) 
    # np.dot(val, weights) # 0.21670189702309739


    ###--------- approach depreciated ------------###
    ###--------- use df to generalize ------------###
    
    ## output xarray.dataarray of points, see examples below
    # this is the way how xarray.Dataset works
    # if you want a xarray.DataArray, first generate a xarray.Dataset, then select DataArray from there.
    #dframe_out = xr.Dataset({col_name: (['points'],var_new)},
    #                        coords={
    ##'time':(['points'],['2002-07-13 00:00:00', '2002-07-22 00:00:00', '2002-07-13 00:00:00']) ,
    #                                'time':(['points'], dframe.time) ,
    #                                'id': (['points'], dframe.id), 
    #                                'lon': (['points'], dframe.lon),
    #                                'lat':(['points'], dframe.lat), 
    #                                'points': (['points'], range(0,len(dframe))) } ) #dims isset to point
    ###-------------------------------------------### 
    #http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    dframe.loc[:, col_name] = pd.Series(var_new, index = dframe.index)
    #print('\n', dframe[col_name])
    return dframe

    
#newtmpAll = sel_points_multilinear(ds_9day, dims = 'points', out ='chlor_a', 
#              time = list(['2002-07-13 00:00:00']), lat = list([5]),  lon = list([70]) )
    

################
##test case 1-4
##### xlat_test = dframe.lat.values + 0.06
#####  xlon_test = dframe.lon.values +0.06


## test case 1: take a single entry (southeast corner for valid values) (passed)
#row_case1 =  pd.DataFrame(data = {'time':'2002-07-13 00:00:00', 'id': 10206,
#                                  'lon':74.7083358765, 'lat':5.20833349228},index=[1])
##print(row_case1)
#ds_9day = xr.open_dataset("./ds_9day.nc")
#result_out1 = sel_points_multilinear(ds_9day, row_case1, dims = 'points', col_name ='chlor_a')  # + 0.06 for both
##print(result_out1)
#recheck the [id], [index], [] & there length in the output


## test case 2: take 3 entries (passed):
#row_case2 =  pd.DataFrame(data = {'time':['2002-07-13 00:00:00', '2002-07-22 00:00:00', '2002-07-13 00:00:00'] ,
#                           'id': [10206, 10206, 10206], 'lon':[74.7083358765, 74.6250076294,74.7083358765],
#                            'lat':[5.20833349228, 5.29166173935, 5.20833349228]}, index=[1,2,3])
##print(row_case2)
#ds_9day = xr.open_dataset("./ds_9day.nc")
#result_out2 = sel_points_multilinear(ds_9day, row_case2, dims = 'points', col_name ='chlor_a')
#print(result_out2)


## test case 3: use the partial real data
# row_case3 = pd.DataFrame(data={'time':list(floatsDFAll_9Dtimeorder.time[:15]),
#                                'lon':list(floatsDFAll_9Dtimeorder.lon[:15]),
#                                'lat':list(floatsDFAll_9Dtimeorder.lat[:15]),
#                                'id':list(floatsDFAll_9Dtimeorder.id[:15]) } )
# print('\n before dropping nan \n', row_case3)
## process to drop nan in any of the columns [id], [lat], [lon], [time]
# row_case3 = row_case3.dropna(subset=['id', 'lat', 'lon', 'time'], how = 'any') # these four fields are critical
# print('\n after dropping nan \n', row_case3)
# result_out3 = sel_points_multilinear(ds_9day, row_case3, dims = 'points', col_name ='chlor_a')
# print('\n after the preprocessing \n', result_out3)
# print('\n this two length should be equal %d == %d?' %(len(row_case3.index), len(result_out3.points) ) )



## test case 4: using the real data
#row_case4 = pd.read_csv("./row_case4.csv")
#ds_9day = xr.open_dataset("./ds_9day.nc")
# bounds
## row_case.lat > ds_9day.lat.min  # descending
## row_case.lat < ds_9day.lat.max  # descending
## row_case.lon > ds_9day.lat.min  # ascending
## row_case.lat > ds_9day.lat.min  # ascending
#result_out4 = sel_points_multilinear(ds_9day, row_case4, dims = 'points', col_name ='chlor_a')
#print('\n after the preprocessing \n', result_out4)
#print('\n this two length should be equal %d >= %d?' %(len(row_case4.index), len(result_out4.points) ) )



#tasks:
# c. check the interface and may need to take the id and index from the float data, decide where to dropna, 
#    perhaps before the functions calls
# d. need to remove or comment out the print outs


# tmpAll = ds_9day.chlor_a.sel_points(time=list(floatsDFAll_9Dtimeorder.time),
#                                     lon=list(floatsDFAll_9Dtimeorder.lon),
#                                     lat=list(floatsDFAll_9Dtimeorder.lat),
#                                     method='nearest')



