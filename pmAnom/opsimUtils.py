import glob
import os
import numpy as np
import pandas as pd
import healpy as hp
import sqlite3
import matplotlib.pyplot as plt
from matplotlib import rcParams
from os.path import splitext, basename

# import rubin_sim.maf python modules
import rubin_sim.maf.metricBundles as metricBundles
import rubin_sim.maf.plots as plots
import rubin_sim.maf.stackers as stackers
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.db as db
import rubin_sim.utils as rs_utils
import rubin_sim


# pre-load run information
run_df = rubin_sim.maf.get_runs()
dataDir = rubin_sim.data.get_data_dir()

# DDF RA/DEC dict
ddfCoord = {
    'COSMOS': (150.11, 2.14),
    'ELAISS1': (9.487, -44.0),
    'XMM-LSS': (35.707, -4.72),
    'ECDFS': (53.15, -28.08),
    'EDFS_a': (58.9, -49.315),
    'EDFS_b': (63.6, -47.6),
    'EDFS': (61.28, -48.42)
}

def show_fbs_dirs():
    """Show available FBS opsim database directories."""
    
    fbs_dirs = glob.glob('/home/idies/workspace/lsst_cadence/FBS_*/')
    
    return fbs_dirs


def show_opsims():
    """Return a list of opsim runs in the current version."""
    return run_df.index.values

    
def fetchPropInfo(dbPath):
    """Get proposal info directly from the sqlite database.
    
    Args:
        dbPath(str): Full path to the opsim sqlite database.
    """
    
    try:
        con = sqlite3.connect(dbPath)
        cursor = con.cursor()
    except Exception as e:
        print(f'fetchPropInfo: {e}')
    
    # get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if ('Proposal',) in tables:
        cursor.execute('Select * from Proposal;')
        propInfo = cursor.fetchall()
        return propInfo
    else:
        print('Proposal infomation not found in database!')
        return None
    
    
def get_ddfNames(opsimdb):
    """Given an opsim database object, return the DDF field names."""
    
#     dbPath = os.path.join(dbDir, run_df.loc[opsimdb, 'filepath'])
    propInfo = fetchPropInfo(opsimdb)
    DD_Ids = [prop[0] for prop in propInfo if prop[2] == 'DD']
    
    DD_names = [propInfo[x][1].split(':')[1] for x in DD_Ids]
    return DD_names


def ddfInfo(opsimdb, ddfName):
    """
    Return DDF metainfo given the name and a opsim database object.

    Args:
        opsimdb: An opsim database object.
        ddfName(str): The name of the requested DDF field, e.g., COSMOS

    Returns:
        ddfInfo(dict): A dictionary containing metainfo (proposalId, RA/DEC and etc.) 
            for the requested DDF field. 
    """

    ddfName = str(ddfName)
    propInfo = fetchPropInfo(opsimdb)
    DD_Ids = [prop[0] for prop in propInfo if prop[2] == 'DD']

    if ddfName not in ddfCoord.keys():
        print('DDF name provided is not correct! Please use one of the below: \n')
        print(list(ddfCoord.keys()))
        return None
    elif len(DD_Ids) == 0:
        print('No DDF in this Opsim run!')
        return None
    else:
        ddfInfo = {}
        ddfInfo['proposalId'] = [prop[0] for prop in propInfo 
                                 if ddfName in prop[1]][0]
        ddfInfo['Coord'] = ddfCoord[ddfName]
        return ddfInfo

def _set_circular_region(ra_center, dec_center, radius, nside=64):
    """function to specify a circular footprint"""
    
    # find the healpixels that cover a circle of radius radius around ra/dec center (deg)
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), 
                         lonlat=True)
    result = np.zeros(len(ra))
    distance = rs_utils._angularSeparation(np.radians(ra_center), 
                                           np.radians(dec_center),
                                           np.radians(ra), np.radians(dec))
    result[np.where(distance < np.radians(radius))] = 1
    return result


def ddf_hp_mask(ddfName, radius, nside):
    """
    Create a healpix slicer for a given DDF.
    
    Args:
        ddfName(str): Name of the DDF.
        radius(float): Raidus of the circular footprint in degree.
        nside(int): Healpix nside.
    """
    
    try:
        ra, dec = ddfCoord[ddfName]
    except:
        raise('DDF not in list!')
    
    return _set_circular_region(ra, dec, radius, nside)


def connect_dbs(dataDir, outDir, dbRuns=None):
    """
    Initiate database objects to all opSim databases in the provided directory.
    Returns a dictionary consisting all database connections and a dictionary
    holding the resultsDb objects.

    Args:
        dataDir(str): The path to the rubin_sim data directory.
        outDir(str): The path to the result database directory.
        dbRuns(list): A list of OpSim runs to connect to.

    Returns:
        opSimDbs(dict): A dictionary containing the OpsimDatabase objects for
            opsim databases in the provided directory, keys are the run names.
        resultDbs(str): A dictionary containing the ResultsDb objects for opsim
            databases in the provided directory, keys are the run names.
    """
    opSimDbs = {}
    resultDbs = {}
    
    if dbRuns is None:
        dbRuns = list(run_df.index)

    for i, dbRun in enumerate(dbRuns):
        dbName = dbRun
        dbPath = os.path.join(dataDir, 'sim_dbs', f'{dbRun}.db')
        opSimDbs[dbName] = db.OpsimDatabase(dbPath)
        resultDbs[dbName] = db.ResultsDb(outDir=outDir, 
                                         database=dbName +'_result.db')
    return (opSimDbs, resultDbs)

def getResultsDbs(resultDbPath):
    """Create a dictionary of resultDb from resultDb files

    Args:
        resultDbPath(str): Path to the directory storing the result databases
            generated by MAF.

    Returns:
        resultDbs(dict): A dictionary containing the ResultDb objects
            reconstructed from result databases in the provided directory.
    """

    resultDbs = {}
    resultDbList = glob.glob(os.path.join(resultDbPath, '*_result.db'))
    for resultDb in resultDbList:
        runName = os.path.basename(resultDb).rsplit('_', 1)[0]
        resultDb = db.ResultsDb(database=resultDb)

        # Don't add empty results.db file,
        if len(resultDb.getAllMetricIds()) > 0:
            resultDbs[runName] = resultDb

    return resultDbs


def bundleDictFromDisk(resultDb, runName, metricDataPath):
    """Load metric data from disk and import them into metricBundles.

    Args:
        resultDb(dict): A ResultDb object.
        runName(str): The name of the opsim database that the metrics stored in
            resultDb was evaluated on.
        metricDataPath(str): The path to the directory where the metric data
            (.npz files) is stored.

    Returns:
        bundleDict(dict): A dictionary of metricBundles reconstructed from data
            stored on disk, the keys designate metric names.
    """

    bundleDict = {}
    displayInfo = resultDb.getMetricDisplayInfo()
    for item in displayInfo:
        metricName = item['metricName']
        metricFileName = item['metricDataFile']
        metricId = item['metricId']
        newbundle = metricBundles.createEmptyMetricBundle()
        newbundle.read(os.path.join(metricDataPath, metricFileName))
        newbundle.setRunName(runName)
        bundleDict[metricId, metricName] = newbundle
    return bundleDict


def get_metricNames(resultDb):
    '''Return the names of metrics stored in the provided resultDb object, the
    names returned are unique regardless of other constrains in the metadata.

    Args:
        resultDb(object): The MAF resultDb object.

    Returns:
        metricNames(list): A list of unique metric names.
    '''

    return list(np.unique(resultDb.getMetricDisplayInfo()['metricName']))


def get_metricMetadata(resultDb, metricName=None, metricId=None):
    '''Print metricMetadata for metrics in bundleDict, if metricName/metricId 
    is provided, will show metrics with the provided name/ID only. 

    Args:
        resultDb(object): The MAF resultDb object.
        metricName(str): The name of a specific metric.
        metricId(int): The ID of a specific metric

    Returns:
        keys: A pandas dataframe listing the requested metadata.
    '''
    metadata = resultDb.getMetricDisplayInfo()
    metadata = metadata[['metricId', 'metricName', 'slicerName', 'sqlConstraint',
                         'metricInfoLabel', 'metricDataFile']]
    if metricId is not None:
        return pd.DataFrame(metadata[metadata['metricId'] == metricId])
    elif metricName is not None:
        return pd.DataFrame(metadata[metadata['metricName'] == metricName])
    else:
        return pd.DataFrame(metadata)


def getSummaryStatNames(resultDb, metricName, metricId=None):
    '''Return the names of computed summary statistic for a particular metric.

    Args:
        resultDb(object): The MAF resultDb object.
        metricName(str): The name of a metric.
        metricId(int): The ID for a metric, if there exists multiple metrics of
            the same name (Optional).

    Returns:
        summaryStatNames(list): A list of summary statistic names.
    '''
    if metricId is not None:
        return {'metricId': metricId,
                'StatNames': list(np.unique(resultDb.getSummaryStats(metricId)['summaryName']))}
    else:
        metricIds = resultDb.getMetricId(metricName=metricName)
        returnList = []
        for metricId in metricIds:
            returnList.append({'metricId': metricId,
                               'StatNames': list(np.unique(
                                   resultDb.getSummaryStats(metricId)['summaryName']))})
        return returnList


def getSummary(resultDbs, metricName, summaryStatName, runNames=None, pandas=True, **kwargs):
    '''
    Return one summary statstic for opsims (included in the resultDbs) on a
    particualr metric given some constraints.

    Args:
        resultDbs(dict): A dictionary of resultDbs, keys are run names.
        metricName(str): The name of the metric to get summary statistic for.
        summaryStatName(str): The name of the summary statistic get (e.g., Median)
        runNames(list): A list of runNames to retrieve summary stats, if not
            all in resultDbs.
        pandas (bool): Whether to return result in pandas dataframe, otherwise a dictionary
            of numpy record arrays.

    Returns:
        stats(dict): Each element is a list of summary stats for the corresponding
            opSim run indicated by the key. This list could has a size > 1, given
            that we can run one metric with different sql constraints.
    '''
    
    if runNames is None:
        runNames = list(resultDbs.keys())
    elif not (set(runNames) <= set(resultDbs.keys())): 
        raise Exception("Provided runNames don't match the record!")
        
    stats = {}
    for run in runNames:
        mIds = resultDbs[run].getMetricId(metricName=metricName, **kwargs)
        stats[run] = np.unique(resultDbs[run].getSummaryStats(
            mIds, summaryName=summaryStatName))

    if pandas:
        df_ls = []
        for key in stats:
            df = pd.DataFrame.from_records(stats[key])
            df['runName'] = key
            df_ls.append(df)
        df_rt = pd.concat(df_ls, ignore_index=True)
        return df_rt
    else:
        return stats


def plotSummaryBar(resultDbs, metricName, summaryStatName, runNames=None, **kwargs):
    '''
    Generate bar plot using summary statistics for comparison between opSims.

    Args:
        resultDbs(dict): A dictionary of resultDb, keys are run names.
        metricName(str): The name of the metric to get summary statistic for.
        summaryStatName(str): The name of the summary statistic get (e.g., Median)
        runNames(list): A list of runNames to plot summary stats, if not all in
            resultDbs.
    '''

    # matplotlib para config
    rcParams['text.usetex'] = False
    rcParams['font.size'] = 10
    rcParams['axes.titlepad'] = 10

    stats = getSummary(resultDbs, metricName,
                       summaryStatName, runNames, pandas=False)

    if runNames is None:
        runNames = list(resultDbs.keys())
    elif not (set(runNames) <= set(resultDbs.keys())): 
        raise Exception("Provided runNames don't match the record!")

    stats_size = stats[runNames[0]].shape[0]
    x = np.arange(len(runNames))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(stats_size):
        label = '{}_{}_{}'.format(
            metricName, stats[runNames[0]]['slicerName'][i],
                stats[runNames[0]]['metricInfoLabel'][i])
        summaryValues = []

        if stats_size == 1:
            shift = 0
        else:
            shift = -width/2 + i*width/(stats_size-1)

        for key in stats:
            try:
                summaryValues.append(stats[key]['summaryValue'][i])
            except IndexError:
                summaryValues.append(0)

        ax.bar(x+shift, summaryValues, width, label=label)

    # set whether to draw hline
    hline = kwargs.get('axhline')
    if hline is not None:
        plt.axhline(int(hline), color='k', ls='--')

    ax.set_xticks(x)
    ax.set_xticklabels(runNames)
    plt.xticks(rotation=80)
    plt.title('Bar Chart for Summary Stat: {} of Metric: {}'.format(
        summaryStatName, metricName))
    plt.ylabel('Summary Values')
    plt.legend(loc='best')
    fig.tight_layout()

    
def plotSummaryBarh(resultDbs, metricName, summaryStatName, runNames=None, **kwargs):
    '''
    Generate horizontal bar plot using summary statistics for comparison between opSims.

    Args:
        resultDbs(dict): A dictionary of resultDb, keys are run names.
        metricName(str): The name of the metric to get summary statistic for.
        summaryStatName(str): The name of the summary statistic get (e.g., Median)
        runNames(list): A list of runNames to plot summary stats, if not all in
            resultDbs.
    '''

    # matplotlib para config
    rcParams['text.usetex'] = False
    rcParams['font.size'] = 12
    rcParams['axes.titlepad'] = 10

    stats = getSummary(resultDbs, metricName,
                       summaryStatName, runNames, pandas=False)

    if runNames is None:
        runNames = list(resultDbs.keys())
    elif not (set(runNames) <= set(resultDbs.keys())):        
        raise Exception("Provided runNames don't match the record!")

    stats_size = stats[runNames[0]].shape[0]
    y = np.arange(len(runNames))
    width = 0.5/stats_size
    
    # compute fig size
    fig_y = len(runNames)*0.8
    fig, ax = plt.subplots(figsize=(12, fig_y))

    for i in range(stats_size):
        label = '{}_{}_{}'.format(
            metricName, stats[runNames[0]]['slicerName'][i],
                stats[runNames[0]]['metricInfoLabel'][i])
        summaryValues = []

        if stats_size == 1:
            shift = 0
        else:
            shift = -width/2 + i*width/(stats_size-1)

        for key in stats:
            try:
                summaryValues.append(stats[key]['summaryValue'][i])
            except IndexError:
                summaryValues.append(0)

        ax.barh(y+shift, summaryValues, width, label=label)

    # set whether to draw vline
    vline = kwargs.get('axhline')
    if vline is not None:
        plt.axvline(int(vline), color='k', ls='--')

    ax.set_yticks(y)
    ax.set_yticklabels(runNames)
#     plt.yticks(rotation=10)
    plt.xlabel('Summary Statistic Value')
    plt.title('Bar Chart for Summary Stat: {} of Metric: {}'.format(
        summaryStatName, metricName))
    plt.legend(loc='best')
    fig.tight_layout()


def plotHist(bundleDicts, metricKey, runNames=None, **kwargs):
    '''
    Plot histogram of evaluated metrics for each opsim in the bundleDicts on
    one figure.

    Args:
        bundleDicts(dict): A dictionary of bundleDict, keys are run names.
        metricKey(tuple): A tuple dictionary key for a specific metric, slicer 
            and constraint combination.
        runNames(list): A list of opsim run names from which the metric values use
            to plot histogram are cacluated, default to None, meaning all opsims.
    '''
    # init handler
    ph = plots.PlotHandler(savefig=False)

    # init plot
    healpixhist = plots.HealpixHistogram()
        
    # option to provide own plotDict for MAF
    if kwargs.get('plotDict') is not None:
        plotDictTemp = kwargs.get('plotDict')
    else:
        plotDictTemp = {'figsize': (8, 6), 'fontsize': 15, 'labelsize': 13}
    
    # check plotting key args
    if kwargs.get('logScale') is not None: plotDictTemp['logScale'] = kwargs.get('logScale')
    
    plotDicts = []
    bundleList = []

    # match keys & remove none
    metricKeys = key_match(bundleDicts, metricKey, **kwargs)
    metricKeys = {key:value for (key, value) in metricKeys.items() if value is not None}
    
    # loop over all opsims
    if runNames is None:
        runNames = list(metricKeys.keys())
    # check if provided runName indeed exists
    elif not (set(runNames) <= set(bundleDicts.keys())):
        raise Exception("Provided runNames don't match the record!")

    for runName in runNames:    
        plotDict = plotDictTemp.copy()
        plotDict.update({'label': runName})
        plotDicts.append(plotDict)
        bundleList.append(bundleDicts[runName][metricKeys[runName]])

    # set metrics to plot togehter
    ph.setMetricBundles(bundleList)
    fn = ph.plot(plotFunc=healpixhist, plotDicts=plotDicts)

    # set whether to draw hline
    vline = kwargs.get('axvline')
    if vline is not None:
        plt.figure(fn)
        plt.axvline(int(vline), color='k', ls='--')


def key_match(bundleDicts, metricKey, src_run=None, resultDbs=None, **kwargs):
    """
    Return metricKeys in all metric bundleDict given a bundleDicts 
    and a metricKeys from one of the bundleDict. If the metricName is 
    not unique and the order of metrics is not consistent across all OpSim runs 
    the name of the source opsim run and the resultDbs dictionary is required.
    
    Args:
        bundleDicts(dict): A dictionary of bundleDict, keys are run names.
        metricKey(tuple): A tuple dictionary key for a specific metric, slicer 
            and constraint combination.
        src_run(str, optional): The opsim run where the provided metricKey come from.
        resultDbs(dict, optional): The dictionary of resultDbs.
    
    Returns:
        metricKeys(dict): A dictionary of matched metricKey tuple, each is indexed
            by the runName.
    """
    
    runs = list(bundleDicts.keys())
    metricKeys = {}
    
    for run in bundleDicts:
        
        # metircNames in the current run
        names = [key[1] for key in bundleDicts[run].keys()]
        
        # if not available, assign none
        if not metricKey[1] in names:
            print(f'No matching metric found in run: {run}! Assigned None.')
            metricKeys[run] = None
        
        # 1st check if keyName unique
        elif (len(names) == len(np.unique(names))):
            keys = [*bundleDicts[run].keys()]
            metricKeys[run] = [elem for elem in keys if elem[1] 
                               == metricKey[1]][0]

        # 2nd check if the order persist across all opsim
        elif all(list(map(lambda x: metricKey in x[1].keys(), bundleDicts.items()))):
            metricKeys[run] = metricKey

        # if neither above, do the brute force search using resultDbs
        elif (src_run is not None) and (src_run in runs) and (resultDbs is not None):
            ref_rows = get_metricMetadata(resultDbs[src_run], metricName=metricKey[1])
            ref_row = ref_rows[ref_rows.metricId == metricKey[0]]
            ref_slicer = ref_row.slicerName.values[0]
            ref_meta = ref_row.metricInfoLabel.values[0]

            runMeta = resultDbs[run].getMetricDisplayInfo()
            mask1 = runMeta['slicerName'] == ref_slicer
            mask2 = runMeta['metricInfoLabel'] == ref_meta
            metricId = runMeta['metricId'][mask1 & mask2][0]

            metricKeys[run] = (metricId, metricKey[1])

        else:
            raise Exception("Can't locate a match with the given information!")

    return metricKeys


def plotSky(bundleDicts, metricKey, **kwargs):
    '''
    Plot healpix skymap for each opSim given a metric. One figure per opSim.

    Args:
        bundleDicts(dict): A dictionary of bundleDict, keys are run names.
        metricKey(tuple): A tuple dictionary key for a specific metric, slicer 
            and constraint combination.
    '''
    
    # init handler, plot, etc.
    ph = plots.PlotHandler(savefig=False)
    healpixSky = plots.HealpixSkyMap()
    metricKeys = key_match(bundleDicts, metricKey, **kwargs) # match keys
    metricKeys = {key:value for (key, value) in metricKeys.items() if value is not None}
    
    # option to provide own plotDict for MAF
    if kwargs.get('plotDict') is not None:
        plotDict = kwargs.get('plotDict')
    else:
        plotDict = {}
           
    for run in metricKeys.keys():
        ph.setMetricBundles([bundleDicts[run][metricKeys[run]]])
        ph.plot(plotFunc=healpixSky, plotDicts=plotDict)

    
    
def plotSky_DDF(mb, ddfName, xsize=250, scale=None):
    '''
    Plot High-Res DDF skymap. 

    Args:
        mb: MetricBundle object.
        ddfName(str): The string name of the DDF field, e.g., COSMOS.
        xsize(int): The dimention of the plot in pixels, default is 250 pixels.
        scale (func): A scaling function for the metric data, e.g., np.log10
    '''
    if scale is None:
        hp.gnomview(mb.metricValues, rot=list(ddfCoord[ddfName]), flip='astro',
                    xsize=xsize)
    else:
        try:
            mbValues = mb.metricValues.copy()
            mask = mbValues.mask
            data = mbValues.data[~mask]
            nData = scale(data)
        except Exception as e:
            print(e)
            return None
        else:
            mbValues.data[~mask] = nData

        hp.gnomview(mbValues, rot=list(ddfCoord[ddfName]), flip='astro',
                    xsize=xsize)

    hp.graticule()
    plt.title(
        f'DDF:{ddfName}, Metric:{mb.metric.name}, RunName:{mb.runName}, Scale:{scale}')