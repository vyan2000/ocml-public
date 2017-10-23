import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import seaborn as sns
#from datetime import datetime
import datetime

def add_chl_rates(floatsDF, result_interpDF):
    """
    function for adding the rate of change of chlor_a and log-scale chlor_a 
    to the dataframe. It is done by taking temporal difference using xarray
    dataset.
    ----------
    input: 
    floatsDF -- pandas.dataframe of floats data
    result_interpDF -- xarray.dataset of chlor_a generated by interpolation routines

    output: 
    combined_dfRate_interpolate -- four columns added {chlor_a, chlor_a_log_e, 
                                   chl_rate, chl_log_e_rate}
    """

    print("\n ******* Adding the chlor_a ******* \n")
    ###### make the interpolation dateframe as a dataframe
    tmpdf_interpolate = result_interpDF.to_dataframe().dropna()  # dropna along chlor_a
    print('shape of the interpolation dateframe', tmpdf_interpolate.shape) # 2373 rows  > 1631(the original approach)
    print(tmpdf_interpolate[:20])   

    # merge(left) floatsDFAll_9Dtimeorder and interpolation dateframe
    combined_df_interpolate = pd.merge(floatsDF, tmpdf_interpolate, on=['id','lat','lon','time'], how='left')

    # check
    print('\n shape (before adding the interpolated chlor_a)', floatsDF.shape)
    print('shape (after added the interpolated chlor_a)\n',combined_df_interpolate.shape)
    check1 = combined_df_interpolate[~np.isnan(combined_df_interpolate.chlor_a)]
    print('comparison of the chlor_a values \n', check1.sort_values(['time','id'],ascending=True).chlor_a[:20])

    def scale(x):
        logged = np.log(x)
        return logged

    #print(floatsAll_timeorder['chlor_a'].apply(scale))
    combined_df_interpolate['chlor_a_log_e'] = combined_df_interpolate['chlor_a'].apply(scale)
    combined_df_interpolate
    #print("after the transformation the nan values in 'chlor_a_log_e' is", 
    # combined_df_interpolate.chlor_a_log_e.isnull().sum() )

    # summaries
    print("\n ******* \n summary of interpolated chlor_a \n", combined_df_interpolate.chlor_a.describe())
    print("\n ******* \n summary of interpolated chlor_a_log_e \n ", combined_df_interpolate.chlor_a_log_e.describe())


    print("\n ******* Take the Diff of chlor_a ******* \n" )
    # prepare the data in dataset and about to take the diff
    # set time & id as the index); use reset_index to revert this operation
    tmp_interpolate = xr.Dataset.from_dataframe(combined_df_interpolate.set_index(['time','id']) ) 
    # take the diff on the chlor_a and the log-scale chlor_a
    chlor_a_rate = tmp_interpolate.diff(dim='time',n=1).chlor_a.to_series().reset_index()
    chlor_a_log_e_rate = tmp_interpolate.diff(dim='time',n=1).chlor_a_log_e.to_series().reset_index()
    # rename the columns
    chlor_a_rate.rename(columns={'chlor_a':'chl_rate'}, inplace='True')
    chlor_a_rate
    chlor_a_log_e_rate.rename(columns={'chlor_a_log_e':'chl_log_e_rate'}, inplace='Trace')
    chlor_a_log_e_rate

    # left-merge the two dataframes {floatsDFAll_XDtimeorder; chlor_a_rate} into
    #  one dataframe based on the index {id, time}
    combined_dfRate_interpolate=pd.merge(combined_df_interpolate, chlor_a_rate, 
                                        on=['time','id'], how = 'left')
    combined_dfRate_interpolate=pd.merge(combined_dfRate_interpolate, chlor_a_log_e_rate, 
                                        on=['time','id'], how = 'left')

    # check 
    print('check the sum of the chlor_a_rate before the merge', chlor_a_rate.chl_rate.sum())
    print('check the sum of the chlor_a_rate after the merge', combined_dfRate_interpolate.chl_rate.sum())

    # check
    print('check the sum of the chlor_a_log_e_rate before the merge', chlor_a_log_e_rate.chl_log_e_rate.sum())
    print('check the sum of the chlor_a_log_e_rate after the merge', combined_dfRate_interpolate.chl_log_e_rate.sum())

    # summaries
    print("\n ******* \n summary of the rate of change of chlor_a \n", 
          combined_dfRate_interpolate.chl_rate.describe())

    # summaries
    print("\n ******* \n summary of the rate of change of log-scale chlor-a \n",
          combined_dfRate_interpolate.chl_log_e_rate.describe())
    
    return combined_dfRate_interpolate


def add_chl_rates_globcolour(floatDF_tmp, freq):
    """
    function for adding the rate of change of chlor_a and log-scale chlor_a 
    to the dataframe. It is done by taking temporal difference using xarray
    dataset; Nondimensionalization for the daily and weekly temporal scales;
    Standardization for the daily scale.
    ----------
    input: 
    floatDF_tmp -- pandas.dataframe of floats data
    freq        -- current resampling frequency

    output: 
    floatDF_tmp -- updated dataframe
                   columns added {chlor_a_log_e, chl_rate, chl_log_e_rate, chl_rate_week,
                   chl_log_e_rate_week, chl_rate_stand, chl_log_e_rate_stand}
    """
    def scale(x):
        logged = np.log(x)
        return logged
    
    # chlor_a on the log-scale
    floatDF_tmp['chlor_a_log_e'] = floatDF_tmp['chlor_a'].apply(scale)
    
    print("\n ******* Take the Diff of chlor_a ******* \n" )
    # prepare the data in dataset and about to take the diff
    # set time & id as the index); use reset_index to revert this operation
    tmp_interpolate = xr.Dataset.from_dataframe(floatDF_tmp.set_index(['time','id']) ) 
    # take the diff on the chlor_a and the log-scale chlor_a
    chlor_a_rate = tmp_interpolate.diff(dim='time',n=1).chlor_a.to_series().reset_index()
    chlor_a_log_e_rate = tmp_interpolate.diff(dim='time',n=1).chlor_a_log_e.to_series().reset_index()
    
    print("\n *** the resampling freqency used for nondimensionalization is %dD *** \n" % freq)
    
    # nondimensionalization -- # per day
    chlor_a_rate['chlor_a'] = chlor_a_rate['chlor_a'].div(freq)
    chlor_a_log_e_rate['chlor_a_log_e'] = chlor_a_log_e_rate['chlor_a_log_e'].div(freq)
    
    # rename the columns
    chlor_a_rate.rename(columns={'chlor_a':'chl_rate'}, inplace='True')
    chlor_a_log_e_rate.rename(columns={'chlor_a_log_e':'chl_log_e_rate'}, inplace='Trace')
    
    # left-merge the two dataframes {floatsDFAll_XDtimeorder; chlor_a_rate} into
    #  one dataframe based on the index {id, time}
    floatDF_tmp=pd.merge(floatDF_tmp, chlor_a_rate, 
                        on=['time','id'], how = 'left')
    floatDF_tmp=pd.merge(floatDF_tmp, chlor_a_log_e_rate, 
                         on=['time','id'], how = 'left')
    
    #  nondimensionalization -- # per week
    floatDF_tmp['chl_rate_week'] = floatDF_tmp['chl_rate'].mul(7.)
    floatDF_tmp['chl_log_e_rate_week'] = floatDF_tmp['chl_log_e_rate'].mul(7.)
    
    # add standardized rate of change
    floatDF_tmp['chl_rate_stand'] = (floatDF_tmp['chl_rate'] - floatDF_tmp['chl_rate'].mean()) / floatDF_tmp['chl_rate'].std()
    floatDF_tmp['chl_log_e_rate_stand'] = (floatDF_tmp['chl_log_e_rate'] - floatDF_tmp['chl_log_e_rate'].mean()) / floatDF_tmp['chl_log_e_rate'].std()
    
    # check 
    print('check the sum of the chlor_a_rate before the merge', chlor_a_rate.chl_rate.sum())
    print('check the sum of the chlor_a_rate after the merge', floatDF_tmp.chl_rate.sum())
    
    # check
    print('check the sum of the chlor_a_log_e_rate before the merge', chlor_a_log_e_rate.chl_log_e_rate.sum())
    print('check the sum of the chlor_a_log_e_rate after the merge', floatDF_tmp.chl_log_e_rate.sum())

    # summaries
    print("\n ******* \n summary of the rate of change of chlor_a \n", 
          floatDF_tmp.chl_rate.describe())

    # summaries
    print("\n ******* \n summary of the rate of change of log-scale chlor-a \n",
          floatDF_tmp.chl_log_e_rate.describe())
    
    print(floatDF_tmp)
    return floatDF_tmp

def spatial_hist_plots_chl_rate(dfRate):
    """
    function for spatial, histogram, and yearly plots of 
    {chlor_a, chlor_a_log_e, chl_rate, chl_log_e_rate} with 
    summary statistics on the features. 
    input:  
    dfRate-- pandas.dataframe contains chlor_a and chl_rate

    output:
    None

    """
    # visualize the float around the arabian sea region
    fig, ax  = plt.subplots(figsize=(12,10))
    dfRate.plot(kind='scatter', x='lon', y='lat', c='chlor_a', cmap='RdBu_r',
                vmin=0, vmax=5, edgecolor='none', ax=ax)
    fig.suptitle("Chl-a", fontsize=12)

    # visualize the float around the arabian sea region
    fig, ax  = plt.subplots(figsize=(12,10))
    dfRate.plot(kind='scatter', x='lon', y='lat', c='chlor_a_log_e', cmap='RdBu_r',
                vmin=-3, vmax=4, edgecolor='none', ax=ax)
    print("Interpolation: valid data point for chlor_a",
          dfRate.chlor_a.dropna().shape)  # (2373,)
    print("Interpolation: valid data point for chlor_a_log_e",
          dfRate.chlor_a_log_e.dropna().shape)  # (2373,)
    print('Interpolation: valid data points for rate of change of chlor_a',
          dfRate.chl_rate.dropna().shape)   # (1838,) >>> (1008,) data points
    print('Interpolation: valid data points for rate of change of log-scale chlor_a',
          dfRate.chl_log_e_rate.dropna().shape)   # (1838,) >>> (1008,) data points
    fig.suptitle("natural log-scale chl-a", fontsize=12)

    plt.show()

    # visualize the chlorophyll rate, it is *better* to visualize at this scale
    fig, ax  = plt.subplots(figsize=(12,10))
    dfRate.plot(kind='scatter', x='lon', y='lat', c='chl_rate', cmap='RdBu_r',
                vmin=-1, vmax=1, edgecolor='none', ax=ax)
    ax.set_title("rate of change of chl-a", fontsize=12)
    #fig.savefig('spatial_rate_of_change_chl_a.png')  # for plot, to be removed!!
    plt.show()

    # visualize the chlorophyll rate, it is *better* to visualize at this scale
    fig, ax  = plt.subplots(figsize=(12,10))
    dfRate.plot(kind='scatter', x='lon', y='lat', c='chl_log_e_rate', cmap='RdBu_r',
                vmin=-1, vmax=1, edgecolor='none', ax=ax)
    ax.set_title("rate of change of log-scale chlor-a", fontsize=12)
    plt.show()


    print("\n ******* Select from November-01 to March-31 ******* \n" )
    pd.to_datetime(dfRate.time)
    type(pd.to_datetime(dfRate.time))
    ts = pd.Series(0, index=pd.to_datetime(dfRate.time) ) # creat a target time series for masking purpose

    # take the month out
    month = ts.index.month 
    # month.shape # a check on the shape of the month.
    selector = ((11==month) | (12==month) | (1==month) | (2==month) | (3==month) )  
    selector
    print('shape of the selector', selector.shape)
    print("all the data count during 'Nov-01 to Mar-31' is",
          dfRate[selector].chl_rate.dropna().shape) # (977,) >>> (672,)
    print('all the data count is',
          dfRate.chl_rate.dropna().shape )   # (1838,) >>> total (1008,)

    print("\n ******* histogram of the rate of change of chl-a ******* \n" )
    ###### histogram for non standarized data
    print("\n ** summary of chl_rate ** \n", dfRate[selector].chl_rate.dropna().describe())
    axfloat = dfRate[selector].chl_rate.dropna().hist(bins=100,range=[-0.5,0.5])
    axfloat.set_title('chl_rate', fontsize = 10)
    plt.show()

    # standarized series
    ts = dfRate[selector].chl_rate.dropna()
    ts_standardized = (ts - ts.mean())/ts.std()
    print("\n ** summary of standardized chl_rate ** \n",ts_standardized.describe())
    axts = ts_standardized.hist(bins=100,range=[-0.5,0.5])
    axts.set_title('standardized chl_rate', fontsize = 10)
    plt.show()

    print("\n ******* subplots of the rate of change of chl-a for different years ******* \n" )
    ###### using all the data from Jan. to Dec.
    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.suptitle("Rate of change of chl-a for different years", fontsize=12)

    print("\n count of data points for each year: \n")
    for i, ax in zip(range(2002,2017), axes.flat) :
        tmpyear = dfRate[ (dfRate.time > str(i))  & (dfRate.time < str(i+1)) ] # if year i
        #fig, ax  = plt.subplots(figsize=(12,10))
        print(tmpyear.chl_rate.dropna().shape)   # total is 1001
        tmpyear.plot(kind='scatter', x='lon', y='lat', c='chl_rate', cmap='RdBu_r',
                     vmin=-0.6, vmax=0.6, edgecolor='none', ax=ax)
        ax.set_title('year %g' % i)         
    # remove the extra figure
    ax = plt.subplot(8,2,16)
    fig.delaxes(ax)
    plt.show()

    ###### using the data from Nov. to Mar.
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.suptitle("Rate of change of chl-a during 'November-March' for different years", fontsize=12)

    print("\n count of data points during 'November-March' for each year: \n")
    for i, ax in zip(range(2002,2016), axes.flat) :
        tmpyear = dfRate[ (dfRate.time >= (str(i)+ '-11-01') )  & (dfRate.time <= (str(i+1)+'-03-31') ) ] # if year i
        # select only particular month, Nov 1 to March 31
        #fig, ax  = plt.subplots(figsize=(12,10))
        print(tmpyear.chl_rate.dropna().shape)  #  (1826,)>>> total (1001,)
        tmpyear.plot(kind='scatter', x='lon', y='lat', c='chl_rate', cmap='RdBu_r',
                     vmin=-0.6, vmax=0.6, edgecolor='none', ax=ax)
        ax.set_title('year %g' % i) 
    plt.show()
    
    return 0


def reduce_to_NovMar(dfRate):
    '''
    fucntions for reducing the dataset to the period 'Nov-01 to March-31'


    :param dfRate:
    :return:
    '''

    df_list = []
    for i in range(2002, 2017):
        tmpyear = dfRate[(dfRate.time >= (str(i) + '-11-01')) & (
                                               dfRate.time <= (str(i + 1) + '-03-31'))]  # if year i
        # select only particular month, Nov 1 to March 31
        df_list.append(tmpyear)

    dfRate_NovMar = pd.concat(df_list)
    print('all the data count in [11-01, 03-31]  is ', dfRate_NovMar.chl_rate.dropna().shape)  # again, total is (977,)

    return dfRate_NovMar



def nondimensionalize_chl_rate(dfRate, freq):
    """
    function for nondimensonalization of {chl_rate, chl_log_e_rate}
    function for generating weekly rate of change
    function for normalized rate of change
    input:
    dfRate -- pandas.dataframe contains {chlor_a, chlor_a_log_e, chl_rate, chl_log_e_rate}
    freq   -- resampling frequency
    output:
    dfRate -- modified, of course
    """
    # non-dimensionalize the rate of change
    dfRate['chl_rate'] = dfRate['chl_rate'].div(freq)
    dfRate['chl_log_e_rate'] = dfRate['chl_log_e_rate'].div(freq)

    # add standardized rate of change
    dfRate['chl_rate_stand'] = (dfRate['chl_rate'] - dfRate['chl_rate'].mean()) / dfRate['chl_rate'].std()
    dfRate['chl_log_e_rate_stand'] = (dfRate['chl_log_e_rate'] - dfRate['chl_log_e_rate'].mean()) / dfRate['chl_log_e_rate'].std()

    # add weekly rate of change
    dfRate['chl_rate_week'] = dfRate['chl_rate'].mul(7.)
    dfRate['chl_log_e_rate_week'] = dfRate['chl_log_e_rate'].mul(7.)


    # print("\n ******  printing out the rates for validation ****** \n ",
    #       dfRate.dropna().sort_values(['id','time'])[:20])
    # also has a routine at the outside for checking

    return dfRate


def add_week(df_input):
    """
    function for adding 'week no.' to the dataset
    week-1 starts from Nov-1
    week-24 end at March-31
             for dropping

    input:
    df_input -- dataframe to work on

    output:
    df_out -- dataframe now has the week no. added
    """

    # convert into datetime
    df_input['time'] = pd.to_datetime(df_input['time'])  # ,format='%m/%d/%y %I:%M%p'
    #print(df_input.sort_values(by=['id', 'time']).head() )  # a check to delete


    '''
    ## check the week numbers of the range from Nov-01-01 to Mar-01-01
    for year in range(2002, 2017):
        print(str(year) + '-11-01 is week', datetime.datetime(year, 11, 1).isocalendar()[1])  # 44, 45,

    print('----')
    for year in range(2002, 2017):
        print(str(year) + '-3-31 is week', datetime.datetime(year, 3, 31).isocalendar()[1])  # 13, 14
    '''

    ###
    # Approach 1 depreciated
    # grouped = df_timed.chl_rate.groupby(df_timed.index.week)
    # grouped.plot.box()

    ###
    # Approach 2
    # prepare data  a. use index or columns to group

    ###
    # select the corresponding weeks, prepare the data
    df_timed = df_input.set_index('time')
    df_timed['week'] = df_timed.index.week

    # now rotate the index to make Nov-01-01 the first month
    print('the min and max of the week index is %d, %d :' % (df_timed.week.min(), df_timed.week.max()))
    # make the 44th week the 1st week
    df_timed['week_rotate'] = (df_timed.week + 10) % 53
    df_timed.week_rotate.describe()

    # take a look
    #print("\n ****** dataset(with nans) with the week no. added ****** \n", df_timed_NovMar[:20])
    print("\n ****** dataset with the week no. added ****** \n", df_timed.dropna()[:20])

    df_out = df_timed.reset_index()
    return df_out

def spatial_plots_chl_rate_weekly(dfRate):
    """
    http://stackoverflow.com/questions/17725927/boxplots-in-matplotlib-markers-and-outliers

    function for spatial, histogram, and weekly plots of {chl_rate, chl_log_e_rate} with
    input:
    dfRate -- pandas.dataframe contains {chlor_a, chlor_a_log_e, chl_rate, chl_log_e_rate}
    freq   -- resampling frequency
    output:
    None

    """

    tmp = dfRate.chl_log_e_rate.dropna()

    #summary
    print("\n ****** summary of chl_log_e_rate ****** \n", tmp.describe())

    # visualize the ROC of log(chl_a) around the arabian sea region
    # visualize the ROC of log(chl_a) around the arabian sea region
    fig, ax = plt.subplots(figsize=(12, 10))
    dfRate.dropna().plot(kind='scatter', x='lon', y='lat', c='chl_log_e_rate', cmap='RdBu_r',
                      vmin=tmp.median() - 1 * tmp.std(), vmax=tmp.max(), edgecolor='none', ax=ax,
                      title='rate of change of the log-scale chl-a')
    plt.show()


    # histogram for non standarized data
    print("\n ****** histogram of chl_log_e_rate ****** \n", tmp.describe())
    # there are very a few small values on the left
    axdf_chl = tmp.hist(bins=100, range=[-0.5, 0.5])
    axdf_chl.set_title('histogram of the rate of change of the log-scale chl-a', fontsize = 10)
    plt.show()


    # standarized series
    tmp = (tmp - tmp.mean()) / tmp.std()
    print("\n ****** histogram of standardized chl_log_e_rate ****** \n", tmp.describe())
    axdf_chl_stdan = tmp.hist(bins=100, range=[-1.5, 1.5])  # there are very a few small values on the left
    axdf_chl_stdan.set_title('histogram of the standardized rate of change of the log-scale chl-a', fontsize = 10)
    plt.show()


    # prepare for plotting
    print("\n check check check \n", dfRate.head())
    mask_NovMar = (dfRate.week <= 14) | (dfRate.week >= 44)
    df_timed_NovMar = dfRate[mask_NovMar].dropna()
    print("\n df_timed_NovMar.head() \n ", df_timed_NovMar.head())

    print("\n ****** weekly plot of chl_log_e_rate_week ****** \n")
    # weekly plot of chl_log_e_rate
    # This is the rate of change on the exponential scale
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)    # The big subplot
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    axes1 = fig.add_subplot(211)
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate_week'].mean().plot(linestyle="-", color='b',
                                                                                      linewidth=1)
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate_week'].quantile(.75).plot(linestyle="--", color='g',
                                                                                     linewidth=0.35)
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate_week'].quantile(.50).plot(linestyle="--", color='r',
                                                                                     linewidth=0.75)
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate_week'].quantile(.25).plot(linestyle="--", color='g',
                                                                                     linewidth=0.35)
    axes1.set_ylim(-2, 1)
    axes1.set_yticks(np.arange(-2, 1, 0.25))
    axes1.set_xticks(np.arange(1, 25, 1))
    #axes1.legend(bbox_to_anchor=(1.10, 1.05))
    axes1.set_xlabel("")
    axes1.set_ylabel("")   
    
    

    # http://pandas.pydata.org/pandas-docs/version/0.19.1/visualization.html
    # http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    axes2 = fig.add_subplot(212)
    axes2 = df_timed_NovMar.boxplot(column='chl_log_e_rate_week', by='week_rotate', ax=axes2)
    axes2.set_ylim(-2, 1)
    axes2.set_yticks(np.arange(-2, 1, 0.25))
    axes2.set_xticks(np.arange(1, 25, 1))
    axes2.set_xlabel("")
    axes2.set_ylabel("")   

    ###  on the big axes
    ax.set_title("weekly data on the rate of change per week of the log-scale $Chl_a$ Concentration", fontsize=12)
    ax.set_xlabel('week', fontsize=12)
    ax.set_ylabel('rate of change of the log-scale $Chl_a$ in $mg/(m^3 \cdot 7days)$', fontsize=12)
    #plt.savefig("box_line_plots_chl_log_scale_rate.png")
    plt.show()
    plt.close()
    
    
    print("\n ****** weekly plot of chl_rate_week ****** \n")
    # weekly plot on the Lagrangian rate of change of the log-scale chl-a
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)    # The big subplot
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    axes1 = fig.add_subplot(211)
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_rate_week'].mean().plot(linestyle="-", color='b',
                                                                                  linewidth=1)
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_rate_week'].quantile(.75).plot(linestyle="--", color='g',
                                                                                 linewidth=0.35)
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_rate_week'].quantile(.50).plot(linestyle="--", color='r',
                                                                                 linewidth=0.75)
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_rate_week'].quantile(.25).plot(linestyle="--", color='g',
                                                                                 linewidth=0.35)
    
    axes1.set_ylim(-2, 3)
    axes1.set_yticks(np.arange(-2, 3, 0.5))
    axes1.set_xticks(np.arange(1, 25, 1))
    #axes1.legend(bbox_to_anchor=(1.10, 1.05))
    axes1.set_xlabel("")
    axes1.set_ylabel("")   
    

    # http://pandas.pydata.org/pandas-docs/version/0.19.1/visualization.html
    # http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    axes2 = fig.add_subplot(212)
    axes2 = df_timed_NovMar.boxplot(column='chl_rate_week', by='week_rotate', ax= axes2)
    axes2.set_ylim(-2, 3)
    axes2.set_yticks(np.arange(-2, 3, 0.5))
    axes2.set_xticks(np.arange(1, 25, 1))
    axes2.set_xlabel("")
    axes2.set_ylabel("")   
    
    ###  on the big axes
    ax.set_title("weekly data on the rate of change per week of the $Chl_a$ Concentration", fontsize=12)
    ax.set_xlabel('week', fontsize=12)
    ax.set_ylabel('rate of change of the $Chl_a$ in $mg/(m^3 \cdot 7days)$', fontsize=12)
    ##################plt.savefig("box_line_plots_chl_rate.png")
    plt.show()
    plt.close()
    
    
    
    
    
    

    print("\n ****** weekly plot of nondimensionalized daily chl_log_e_rate ****** \n")
    # weekly plot of chl_log_e_rate
    # This is the rate of change on the exponential scale
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate'].mean().plot(linestyle="-", color='b',
                                                                                      linewidth=1)
    df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate'].quantile(.75).plot(linestyle="--", color='g',
                                                                                     linewidth=0.35)
    df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate'].quantile(.50).plot(linestyle="--", color='r',
                                                                                     linewidth=0.75)
    df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate'].quantile(.25).plot(linestyle="--", color='g',
                                                                                     linewidth=0.35)
    axes1.set_ylim(-1, 0.5)
    axes1.set_title("Line plot of the weekly data on the rate of change per day of the log-scale $Chl_a$ Concentration",
                    fontsize=10)
    plt.xlabel('week', fontsize=10)
    plt.ylabel('rate of change of the log-scale $Chl_a$ in $mg/(m^3 \cdot day)$', fontsize=10)
    plt.yticks(np.arange(-1, 0.5, 0.25))
    plt.xticks(np.arange(1, 25, 1))
    plt.show()

    # http://pandas.pydata.org/pandas-docs/version/0.19.1/visualization.html
    # http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    axes2 = df_timed_NovMar.boxplot(column='chl_log_e_rate', by='week_rotate')
    plt.suptitle("")  # equivalent
    axes2.set_ylim(-1, 0.5)
    axes2.set_title("Box plot of the weekly data on the rate of change per day of the log-scale $Chl_a$ Concentration",
                    fontsize=10)
    plt.xlabel('week', fontsize=10)
    plt.ylabel('rate of change of the log-scale $Chl_a$ in $mg/(m^3 \cdot day)$', fontsize=10)
    plt.show()


    print("\n ****** weekly plot of nondimensionalized daily chl_rate ****** \n")
    # weekly plot on the Lagrangian rate of change of the log-scale chl-a
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_rate'].mean().plot(linestyle="-", color='b', linewidth=1)
    df_timed_NovMar.groupby(['week_rotate'])['chl_rate'].quantile(.75).plot(linestyle="--", color='g', linewidth=0.35)
    df_timed_NovMar.groupby(['week_rotate'])['chl_rate'].quantile(.50).plot(linestyle="--", color='r', linewidth=0.75)
    df_timed_NovMar.groupby(['week_rotate'])['chl_rate'].quantile(.25).plot(linestyle="--", color='g', linewidth=0.35)
    axes1.set_ylim(-1, 0.5)
    axes1.set_title("Line plot of the weekly data on the rate of change per day of the $Chl_a$ Concentration",
                    fontsize=10)
    plt.xlabel('week', fontsize=10)
    plt.ylabel('rate of change of the $Chl_a$ in $mg/(m^3 \cdot day)$', fontsize=10)
    plt.yticks(np.arange(-1, 0.5, 0.25))
    plt.xticks(np.arange(1, 25, 1))
    plt.show()

    # http://pandas.pydata.org/pandas-docs/version/0.19.1/visualization.html
    # http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    axes2 = df_timed_NovMar.boxplot(column='chl_rate', by='week_rotate')
    plt.suptitle("")  # equivalent
    axes2.set_ylim(-1, 0.5)
    axes2.set_title("Box plot of the weekly data on the rate of change per day of the $Chl_a$ Concentration",
                    fontsize=10)
    plt.xlabel('week', fontsize=10)
    plt.ylabel('rate of change of the $Chl_a$ in $mg/(m^3 \cdot day)$', fontsize=10)
    plt.show()






    print("\n ****** weekly plot of standardized nondimensionalized-daily chl_log_e_rate ****** \n")
    # weekly plot of chl_log_e_rate
    # This is the rate of change on the exponential scale
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate_stand'].mean().plot(linestyle="-", color='b',
                                                                                   linewidth=1)
    df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate_stand'].quantile(.75).plot(linestyle="--", color='g',
                                                                                  linewidth=0.35)
    df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate_stand'].quantile(.50).plot(linestyle="--", color='r',
                                                                                  linewidth=0.75)
    df_timed_NovMar.groupby(['week_rotate'])['chl_log_e_rate_stand'].quantile(.25).plot(linestyle="--", color='g',
                                                                                  linewidth=0.35)
    axes1.set_ylim(-2, 1)
    axes1.set_title("Line plot of the weekly data on the standardized rate of change per day of the "
                    "log-scale $Chl_a$ Concentration",
                    fontsize=10)
    plt.xlabel('week', fontsize=10)
    plt.ylabel('rate of change of the log-scale $Chl_a$ in $mg/(m^3 \cdot day)$', fontsize=10)
    plt.yticks(np.arange(-2, 1, 0.25))
    plt.xticks(np.arange(1, 25, 1))
    plt.show()

    # http://pandas.pydata.org/pandas-docs/version/0.19.1/visualization.html
    # http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    axes2 = df_timed_NovMar.boxplot(column='chl_log_e_rate_stand', by='week_rotate')
    plt.suptitle("")  # equivalent
    axes2.set_ylim(-3, 2)
    axes2.set_title("Box plot of the weekly data on the standardized rate of change per day of the"
                    " log-scale $Chl_a$ Concentration",
                    fontsize=10)
    plt.xlabel('week', fontsize=10)
    plt.ylabel('rate of change of the log-scale $Chl_a$ in $mg/(m^3 \cdot day)$', fontsize=10)
    plt.show()

    print("\n ****** weekly plot of standardized nondimensionalized-daily chl_rate ****** \n")
    # weekly plot on the Lagrangian rate of change of the log-scale chl-a
    axes1 = df_timed_NovMar.groupby(['week_rotate'])['chl_rate_stand'].mean().plot(linestyle="-", color='b',
                                                                                   linewidth=1)
    df_timed_NovMar.groupby(['week_rotate'])['chl_rate_stand'].quantile(.75).plot(linestyle="--", color='g',
                                                                                  linewidth=0.35)
    df_timed_NovMar.groupby(['week_rotate'])['chl_rate_stand'].quantile(.50).plot(linestyle="--", color='r',
                                                                                  linewidth=0.75)
    df_timed_NovMar.groupby(['week_rotate'])['chl_rate_stand'].quantile(.25).plot(linestyle="--", color='g',
                                                                                  linewidth=0.35)
    axes1.set_ylim(-2, 1)
    axes1.set_title("Line plot of the weekly data on the standardized rate of change per day of the"
                    " $Chl_a$ Concentration",
                    fontsize=10)
    plt.xlabel('week', fontsize=10)
    plt.ylabel('rate of change of the $Chl_a$ in $mg/(m^3 \cdot day)$', fontsize=10)
    plt.yticks(np.arange(-2, 1, 0.5))
    plt.xticks(np.arange(1, 25, 1))
    plt.show()

    # http://pandas.pydata.org/pandas-docs/version/0.19.1/visualization.html
    # http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    axes2 = df_timed_NovMar.boxplot(column='chl_rate_stand', by='week_rotate')
    plt.suptitle("")  # equivalent
    axes2.set_ylim(-1, 0.5)
    axes2.set_title("Box plot of the weekly data on the standardized rate of change per day of the"
                    " $Chl_a$ Concentration",
                    fontsize=10)
    plt.xlabel('week', fontsize=10)
    plt.ylabel('rate of change of the $Chl_a$ in $mg/(m^3 \cdot day)$', fontsize=10)
    plt.show()


    print("\n ****** spatial plot of standardized nondimensionalized-daily chl_rate ****** \n")

    df_timed_NovMar_ind = df_timed_NovMar.set_index(['time'])
    # spatial plot for different months --  totally five months 1, 2, 3, 11, 12,
    for i in range(0, 5, 1):
        month_ind = np.array([11, 12, 1, 2, 3])
        month_names = ['November', 'December', 'January', 'February', 'March']
        aa = df_timed_NovMar_ind[df_timed_NovMar_ind.index.month == month_ind[i]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ##aa.plot(kind='scatter', x='lon', y='lat', c='chl_rate', cmap='RdBu_r',
        # vmin=aa.chl_rate.median()-0.5*aa.chl_rate.std(), vmax=aa.chl_rate.median()-0.5*aa.chl_rate.std(),
        # edgecolor='none', ax=ax, title = 'rate of change of the $Chl_a$')
        ##aa.plot(kind='scatter', x='lon', y='lat', c='chl_rate', cmap='RdBu_r',
        # vmin=aa.chl_rate.mean()-0.5*aa.chl_rate.std(), vmax=aa.chl_rate.mean()+0.5*aa.chl_rate.std(),
        # edgecolor='none', ax=ax, title = 'rate of change of the $Chl_a$')
        print('\n\n summary of the Chl_rate \n', aa.chl_rate_week.describe())
        aa.plot(kind='scatter', x='lon', y='lat', c='chl_rate_stand', cmap='RdBu_r',
                vmin=-0.6, vmax=0.6, edgecolor='none',
                ax=ax)
        ax.set_title("Rate of change of the standardized nondimensionalized-daily "
                     "$Chl_a$ in %s" % (month_names[i]), fontsize=10)
        plt.xticks(np.arange(45, 80, 2.5))
        plt.yticks(np.arange(0, 28, 2.5))
        plt.show()

    plt.close('all')


    return 0


def output_chl_rates_dist(infile_df, freq):
    """
    functions for output the dataset as an cvs file for further processing

    input:
    infile_df -- dataframe for input
    freq -- resampling frequency used for generating the output filename

    output:
    None
    """

    print("\n ****** generating the output file and validating the data file ****** \n")

    outfile_name = 'df_chl_dist_out_' + str(freq) + 'D'+'_modisa.csv'

    df_list = []
    for i in range(2002, 2017):
        tmpyear = infile_df[(infile_df.time >= (str(i) + '-11-01')) &
                            (infile_df.time <= (str(i + 1) + '-03-31'))]  # if year i
        # select only particular month, Nov 1 to March 31
        df_list.append(tmpyear)

    df_tmp = pd.concat(df_list)

    print("all the data count during 'Nov-01 to Mar-31' is ", df_tmp.chl_rate.dropna().shape)  # (977,)>>> (692,)
    df_chl_dist_out = df_tmp[~df_tmp.chl_rate.isnull()]  # only keep the non-nan values

    # output to a csv or hdf file
    print(df_chl_dist_out.head())
    # make it specific for the index name
    df_chl_dist_out.index.name = 'index'
    # CSV CSV CSV CSV with specfic index
    df_chl_dist_out.to_csv(outfile_name, sep=',', index_label='index')
    # load CSV output
    test = pd.read_csv(outfile_name, index_col='index')
    print('\n \n ****** the two dataset shoud be the same before and after generating output******\n \n')
    print(test.head())

    return 0
