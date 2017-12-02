import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns



def floatIdBranchOutPrint(df):
    """
    Given a dataframe contains float ids as input, identify the 
    count of appearence of each float in each cycle from 
    Nov 1 to Dec 31
    ==========================================================
    df - input dateframe
    """
    df_tmp = df
    id_set = df_tmp.id.unique()
    for f in id_set:
        count = 0
        for year in range(2002,2018):
            mask = (f == df_tmp.id) & (df_tmp.time >= (str(year) + '-11-01') ) & (df_tmp.time <= (str(year + 1) + '-03-31') )
            if (len(df_tmp[mask].chlor_a) > 2) :
                count = count+1
                if count >1:
                    #print(df_tmp[mask].id)
                    pass
        print("id ", f, " counts ", count)
    return 0


def floatIdBranchOut(df):
    """
    Given a dataframe contains float ids as input, identify the 
    count of appearence of each float in each cycle from 
    Nov 1 to Dec 31. Once a float appear in two cycles, use derived
    float ids to make sure the data of this float during each different
    cycle will have a unique derivated id.
    Formula:  new_id = old_id + 0.05 * (count-1)
    ==========================================================
    df - input dateframe
    """
    df_tmp = df
    id_set = df_tmp.id.unique()
    for f in id_set:
        count = 0
        for year in range(2002,2018):
            mask = (f == df_tmp.id) & (df_tmp.time >= (str(year) + '-11-01') ) & (df_tmp.time <= (str(year + 1) + '-03-31') )
            if (len(df_tmp[mask].chlor_a) > 2) :
                count = count+1
                if count >1:
                    df_tmp.loc[mask, 'id'] = df_tmp.loc[mask,'id']+ 0.05* (count-1)
        #print("id ", f, " counts ", count)
    return df_tmp



def floatIdInterpolationOrSplitting(floatsDF_NovMar_ID, freq):
    """
    Given a dataframe with float ids as input, the float id. For each float id, considering  
    the ratio of nans over the first_valid_index to the last_valid_index, one decide to choose
    interpolation or splitting.
    For interpolation, it is done first on the key variable chlor_a, then use fill forward and
    fill backward for all other variables.
    For splitting, it is done by using a loop to identify all the contiguous non-nan
    subsequences. Each subsequences with have a unique float id derived based on the 
    original float id
    Formular:  new_id = old_id + 0.03 * (count)
    ==========================================================
    df - input dateframe
    freq - resampling frequency; used to determine the time delta
    """
    id_set3 = floatsDF_NovMar_ID.id.unique()  

    # necessary for selection
    floatsDF_NovMar_ID.time = pd.to_datetime(floatsDF_NovMar_ID.time) 

    from datetime import timedelta
    # temp kept, but is not touched, to be dropped
    col_listLDS = ['time', 'id', 'lat', 'lon', 'temp', 've', 'vn', 'spd', 
                   'chlor_a', 'dist', 'cdm', 'kd490', 't865',
                   'par', 'sst4']

    # temp kept, but is not touched, to be dropped
    col_listLDS_subset = ['ve', 'vn', 'spd',
                          'chlor_a', 'dist', 'cdm', 'kd490', 't865',
                          'par', 'sst4']

    floatsDF_sub_list = [] 

    # test case 1:
    # float id == 71140
    # nan's ratio: 0.5
    # split series and series end with two good values 
    #floatsDF_81828 = floatsDF_NovMar_ID[floatsDF_NovMar_ID.id == 71140]

    # test case 3:
    # float id == 81828, 
    # nan's ratio: 0.61, choose threshold 0.62
    # series no need to split
    # floatsDF_81828 = floatsDF_NovMar_ID[floatsDF_NovMar_ID.id == 81828]

    # test case 2:
    # float id == 81828, 
    # nan's ratio: 0.61, choose threshold 0.47
    # split series and series end with one good values

    for i in id_set3:    
        floatsDF_i = floatsDF_NovMar_ID[floatsDF_NovMar_ID.id == i]
        #print(floatsDF_i.set_index("time").chlor_a)
        #print('first valid index:', floatsDF_i.set_index("time").chlor_a.first_valid_index())
        #print('last valid index:',floatsDF_i.set_index("time").chlor_a.last_valid_index())
        # drop everything outside this range
        head =  floatsDF_i.set_index("time").chlor_a.first_valid_index()
        tail =  floatsDF_i.set_index("time").chlor_a.last_valid_index()
        mask = (floatsDF_i.time >= head) & (floatsDF_i.time <= tail)
        floatsDF_i = floatsDF_i[mask]
        print("-----\n head and tail of the nonnan series are identified \n")
        #print(floatsDF_i.set_index("time").chlor_a)

        # calculate the ratio of nans in the trimmed chlor_a series
        length = len(floatsDF_i.chlor_a)
        #print("length is", length)  # all length zero sequentia already be removed!!! before this step
        ratio = floatsDF_i.chlor_a.isnull().sum()/length
        print("the float id is:", floatsDF_i.id.values[0])
        print("the ratio of nans in the trimmed chlor_a series", ratio)

        # if nan's ratio is less than 0.47, then we interpolate the values in between
        if ratio <= 0.47:
            # df.interpolate()
            floatsDF_i['chlor_a'].interpolate(method='linear', axis=0, 
                                              limit=None, inplace=True)
            ###floatsDF_81828['chlor_a'].interpolate(method='linear', axis=0, limit=None, 
            ###inplace=False).plot()
            #floatsDF_i.plot(x='time', y ='chlor_a', title=('id - %.2f' % i))
            #print(floatsDF_i[col_listLDS ])
            ### process all other variables, here we assume other variable are linear...
            # + easy approach: linear interpolation, then fill forward, then fill backward
            # + more dedicated approach: check ratio of each variables
            print("the easy approach: linear interpolation on other var, then ffill, then bfill")
            #this oneliner doesnot work:
            ###floatsDF_81828[col_listLDS_subset].interpolate(method='linear', axis=0, 
            ###limit=None, inplace=True)
            for col in col_listLDS_subset:
                floatsDF_i[col].interpolate(method='linear', axis=0, limit=None, inplace=True)
                floatsDF_i[col_listLDS_subset].fillna(method='ffill')
                floatsDF_i[col_listLDS_subset].fillna(method='bfill')
            print("------ after preprocess completed ------")
            print("------ after preprocess completed ------")
            print("------ after preprocess completed ------")
            #print(floatsDF_i[col_listLDS ])
            
            # there might still be gaps in the interpolated float at this step
            # so we treat the interpolated float as a fresh one.
            #mask = (floatsDF_i.time >= head) & (floatsDF_i.time <= tail) 
            #tmp = floatsDF_i[mask]
            #floatsDF_sub_list.append(tmp)
            
    
    
        ## - if nan's ratio is bigger than 0.47,
        ## - or the float is already been interpolated during the case where nan's ratio < 0.47
        ## - then we split the series as continuous sub-series
        ## while loop
        ## if ratio > 0.47:
        ## loop through the time
        ## https://stackoverflow.com/questions/16782682/timedelta-is-not-defined
    
        # task: collect all the contiguous subsequences in the time series
        # + use time t to loop through the series
        # + start with subhead = t
        # + use flag = 1 to indicate that an subtail is expected
        # + condition branched on whether chlor_a at time t is nan    
        print("head is", head)
        t = head
        print("t is", t)
        flag = 1     # subtail is expected
        subhead = t  # start with a good value
        count = 0    # counter for the subseries in the splitting

        while (t <= tail ):
            bool_array = floatsDF_i[floatsDF_i.time== t].chlor_a.isnull()
        
            if len(bool_array)==0: # there are gaps in data!!!! supposed to be a length 1 array
                detect_nan = True # end of series reached, since there is a gap in data!!!!
            else:
                detect_nan = bool_array.values[0]
            
            
            if (1 == flag) & (~detect_nan ):
                if (tail == t): # end of series reached
                    count = count + 1
                    flag = 10000
                    subtail = t # end of series reached
                    mask = (floatsDF_i.time >= subhead) & (floatsDF_i.time <= subtail) 
                    tmp = floatsDF_i[mask]
                    tmp.loc[:,'id'] = tmp.loc[:,'id'] + 0.03*(count) # hash the float id
                    tmp.plot(x='time', y ='chlor_a', title=('subseries from id - %.2f' % tmp.id.values[0]))
                    floatsDF_sub_list.append(tmp)
                else: 
                    pass
        
            if (1 == flag) & (detect_nan ):
                count = count + 1
                flag = 0 
                subtail = t - timedelta(freq)
                mask = (floatsDF_i.time >= subhead) & (floatsDF_i.time <= subtail) 
                tmp = floatsDF_i[mask]
                tmp.loc[:,'id'] = tmp.loc[:,'id'] + 0.03*(count) # hash the float id
                tmp.plot(x='time', y ='chlor_a', title=('subseries from id - %.2f' % tmp.id.values[0]))
                floatsDF_sub_list.append(tmp)
            
            if (0 == flag) & (detect_nan ):
                pass
        
            if (0 == flag) & (~detect_nan):
                if (tail == t): # end of series reached
                    count = count + 1
                    flag = 10000
                    subhead = t
                    subtail = t
                    mask = (floatsDF_i.time >= subhead) & (floatsDF_i.time <= subtail) 
                    tmp = floatsDF_i[mask]
                    tmp.loc[:,'id'] = tmp.loc[:,'id'] + 0.03*(count) # hash the float id
                    tmp.plot(x='time', y ='chlor_a', title=('subseries from id - %.2f' % tmp.id.values[0]))
                    floatsDF_sub_list.append(tmp)
                else:          # end of series not reached
                    flag = 1
                    subhead = t
        
            t = t + timedelta(freq) # increment on t
    
            #print("now t is", t)

    floatsDF_list = pd.concat(floatsDF_sub_list)
    return floatsDF_list
