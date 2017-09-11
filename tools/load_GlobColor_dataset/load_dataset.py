import xarray as xr
import pandas as pd
import os
from tqdm import tqdm


def load_chl1():
    """
    function for loading GlobColor CHL1 dataset 
    """
    varname = 'chl1'
    dirnameprefix1 = '/Users/vyan2000/work_linux/2Archive/myproject/'
    dirnameprefix2 = '20161024xray_oceancolor/ocean_color-master/data_globcolour/665648402.data/'
    dirname = dirnameprefix1 + dirnameprefix2 + varname + '/chl1_AVW/'
    files = os.listdir(dirname)

    datasets = []
    for name in tqdm(files):
        # load each
        nametime = name[4:12]  # extract time, add specific'time' coordinate, concatenate 
        #print(nametime)
        path = dirname + name
        dset = xr.open_dataset(path, drop_variables = ['CHL1_flags','CHL1_error'])
        dset_new = dset.assign_coords(time=pd.to_datetime(nametime))
        datasets.append(dset_new) 

    dataset_chl = xr.concat(datasets, dim = 'time')
    print(dataset_chl)
    
    return dataset_chl.rename({'CHL1_mean':'chlor_a'})


def load_cdm():
    """
    function for loading Globcolor CDM dataset
    """
    varname = 'CDM'
    dirnameprefix1 = '/Users/vyan2000/work_linux/2Archive/myproject/'
    dirnameprefix2 = '20161024xray_oceancolor/ocean_color-master/data_globcolour/665648402.data/'
    dirname = dirnameprefix1 + dirnameprefix2 + varname + '/'
    files = os.listdir(dirname)

    datasets = []
    for name in tqdm(files):
        # load each
        nametime = name[4:12]  # extract time, add specific'time' coordinate, concatenate 
        #print(nametime)
        path = dirname + name
        dset = xr.open_dataset(path, drop_variables = ['CDM_flags','CDM_error'])
        dset_new = dset.assign_coords(time=pd.to_datetime(nametime))
        datasets.append(dset_new) 

    dataset_cdm = xr.concat(datasets, dim = 'time')
    print(dataset_cdm)    

    return dataset_cdm.rename({'CDM_mean':'cdm'})


def load_kd490():
    """
    function for loading GlobColor kd490 dataset
    """
    varname = 'T865'
    dirnameprefix1 = '/Users/vyan2000/work_linux/2Archive/myproject/'
    dirnameprefix2 = '20161024xray_oceancolor/ocean_color-master/data_globcolour/665648402.data/'
    dirname = dirnameprefix1 + dirnameprefix2 + varname + '/'
    files = os.listdir(dirname)

    datasets = []
    for name in tqdm(files):
        # load each
        nametime = name[4:12]  # extract time, add specific'time' coordinate, concatenate 
        #print(nametime)
        path = dirname + name
        dset = xr.open_dataset(path, drop_variables = ['KD490-LEE_flags'])
        dset_new = dset.assign_coords(time=pd.to_datetime(nametime))
        datasets.append(dset_new) 

    dataset_kd490 = xr.concat(datasets, dim = 'time')
    print(dataset_kd490)    

    return dataset_kd490.rename({'KD490-LEE_mean':'kd490'})

    
def load_par():
    """
    function for loading GlobColor PAR dataset
    """
    varname = 'PAR'
    dirnameprefix1 = '/Users/vyan2000/work_linux/2Archive/myproject/'
    dirnameprefix2 = '20161024xray_oceancolor/ocean_color-master/data_globcolour/665648402.data/'
    dirname = dirnameprefix1 + dirnameprefix2 + varname + '/'
    files = os.listdir(dirname)

    datasets = []
    for name in tqdm(files):
        # load each
        nametime = name[4:12]  # extract time, add specific'time' coordinate, concatenate 
        #print(nametime)
        path = dirname + name
        dset = xr.open_dataset(path, drop_variables = ['PAR_flags','PAR_error'])
        dset_new = dset.assign_coords(time=pd.to_datetime(nametime))
        datasets.append(dset_new) 

    dataset_par = xr.concat(datasets, dim = 'time')
    print(dataset_par)    

    return dataset_par.rename({'PAR_mean':'par'})

   

def load_t865():
    """
    function for loading GlobColor T865 dataset
    """
    varname = 'T865'
    dirnameprefix1 = '/Users/vyan2000/work_linux/2Archive/myproject/'
    dirnameprefix2 = '20161024xray_oceancolor/ocean_color-master/data_globcolour/665648402.data/'
    dirname = dirnameprefix1 + dirnameprefix2 + varname + '/'
    files = os.listdir(dirname)

    datasets = []
    for name in tqdm(files):
        # load each
        nametime = name[4:12]  # extract time, add specific'time' coordinate, concatenate 
        #print(nametime)
        path = dirname + name
        dset = xr.open_dataset(path, drop_variables = ['T865_flags','T865_error'])
        dset_new = dset.assign_coords(time=pd.to_datetime(nametime))
        datasets.append(dset_new) 

    dataset_t865 = xr.concat(datasets, dim = 'time')
    print(dataset_t865)    

    return dataset_t864.rename({'T865_mean':'t865'})



# perhaps load distance to coast here?


