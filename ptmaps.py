import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import geopandas as gpd
import re
from pysal.esda import mapclassify as mc
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

    
def loadContinentalUSShapes(prefix="./"):
    # shapefile
    usc = gpd.read_file(prefix + 'cb_2016_us_county_500k.shp')
    usbox = Polygon([(-125, 24), (-66, 24), (-66, 50), (-125, 50)])
    usc['continental'] = [usbox.contains(x.geometry) for ii, x in usc.iterrows()]
    continental_us = usc[usc.continental]
    return continental_us
    

def loadUSdata(prefix="./"):
    
    # county data with land area (for calculating population density)
    popdens = pd.read_csv(prefix + "DEC_10_SF1_GCTPH1.US05PR_with_ann.csv", encoding = "ISO-8859-1")
    popdens = popdens.drop(popdens.index[0])
    
    # more recent county level population estimates
    popestimates = pd.read_csv(prefix + "PEP_2016_PEPANNRES_with_ann.csv", encoding = "ISO-8859-1")
    popestimates = popestimates.drop(popestimates.index[0])
    popdens_merged = popdens.merge(popestimates, left_on='GCT_STUB.target-geo-id2', right_on='GEO.id2', 
                                   how='left')
    
    continental_us = loadContinentalUSShapes(prefix=prefix)
    # merge in shapefiles, keeping only continental US
    cont_us_popdens = continental_us.merge(popdens_merged, left_on='GEOID', right_on='GCT_STUB.target-geo-id2',
                                           how='left')
    cont_us_popdens['population'] = cont_us_popdens['respop72016'].apply(
        lambda x: float(re.sub(r'\(.+','',x)) if pd.notnull(x) else np.NaN
    )

    # calculate the population density using the 2016 estimate
    cont_us_popdens['popdens'] = [(x.population / float(x.SUBHD0303)) if (pd.notnull(x.population) and pd.notnull(x.SUBHD0303)) else np.NaN 
                                  for ii, x in cont_us_popdens.iterrows()]
    
    cont_us_popdens['area'] = cont_us_popdens['SUBHD0303'].apply(lambda x: float(x) if pd.notnull(x) else x)
    
    statedata = cont_us_popdens.loc[:,['STATEFP', 'population', 'area']].groupby('STATEFP').sum()
    statedata.rename(columns={'population': 'statepop', 'area': 'statearea'}, inplace=True)
    cont_us_popdens = cont_us_popdens.merge(statedata.reset_index(), on='STATEFP')
    return cont_us_popdens
    

def genSmoothedRatio(dat, num, den, num0, den0, alpha, multiplybyval=1, multiplybycol=None):
    return [
        (multiplybyval if multiplybycol is None else x[multiplybycol]) * 
        ((x[num] if (pd.notnull(x[num]) and pd.notnull(x[den])) else 0) + alpha * x[num0]) / 
        ((x[den] if (pd.notnull(x[num]) and pd.notnull(x[den])) else 0) + alpha * x[den0]) for ii, x in dat.iterrows()
    ]


def genSmoothedPopDens(usdat, shrinkagefactor):
    return genSmoothedRatio(usdat, 'population', 'area', 'statepop', 'statearea', alpha=shrinkagefactor)
  
    
def genSmoothedPTP1K(ptdat, shrinkagefactor):
    return genSmoothedRatio(ptdat, 'npts', 'population', 'statepts', 'statepop', alpha=shrinkagefactor, multiplybyval=1000)


def genSmoothedNPTS(ptdat, shrinkagefactor):
    return genSmoothedRatio(ptdat, 'npts', 'population', 'statepts', 'statepop', alpha=shrinkagefactor, multiplybycol='population')
    

# silly little function to generate the legend labels for quantile bins
def get_quantile_labels(nquantiles):
    suffixes={'0': 'th',
              '1': 'st',
              '2': 'nd',
              '3': 'rd',
              '4': 'th',
              '5': 'th',
              '6': 'th',
              '7': 'th',
              '8': 'th',
              '9': 'th',
             }
    
    breaks = np.linspace(0,100, nquantiles + 1)
    break_suffixes = []
    for ii in breaks:
        break_suffixes.append(suffixes[str(int(ii))[-1]])
        
    strings = []
    for ii in range(nquantiles):
        strings.append("{}-{}{} percentile".format(str(int(breaks[ii])), str(int(breaks[ii+1])), break_suffixes[ii+1]))
    return breaks, strings    
    
    
# main choropleth function
def draw_choropleth(mapdat, column, mincol=163, maxcol=226, nquantiles=5, fname=None, scheme='quantiles'):
    cmap = sns.diverging_palette(mincol, maxcol, s=75, l=55, as_cmap=True, center='dark')
    cmap_discrete = sns.diverging_palette(mincol, maxcol, s=99, l=60, n=nquantiles, center='dark')
    
    plt.subplots(figsize=(16, 8))
    ax = plt.subplot2grid((8,16), (0,0), rowspan=8, colspan=14)
    
    mapdat[pd.notnull(mapdat[column])].plot(ax=ax, column=column, cmap=cmap, scheme=scheme, k=nquantiles)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    rowspan = 2 if nquantiles<=5 else 4
    ax2 = plt.subplot2grid((8,16), (4,14),rowspan=rowspan)
    vals, labs = get_quantile_labels(nquantiles)
    dat = pd.DataFrame({'vals': vals[1:]})
    sns.heatmap(dat, square=True, cbar=False, cmap=cmap_discrete, xticklabels=False, yticklabels=labs, ax=ax2)
    ax2.invert_yaxis()
    ax2.tick_params(length=0)
    ax2.yaxis.tick_right()
    plt.yticks(rotation='horizontal')
    if fname:
        plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.show()
    
    
def foldInPTdata(usdat, prefix = "./"):
    uspt = pd.read_csv(prefix + "us_pt_locations.csv")
    
    # create a shapely point for each PT practice
    usptgeo = gpd.GeoDataFrame({'count': uspt['count(1)'],
                                'geometry': [Point(x.longitude, x.latitude) for ii, x in uspt.iterrows()]})
    
    continental_us = loadContinentalUSShapes(prefix=prefix)
    
    # do a spatial join to associate each practice with its US county
    usptmerged = gpd.sjoin(usptgeo, continental_us, how='left', op='intersects' )
    
    # aggregate PT counts by county
    usptbycounty = usptmerged.groupby('GEOID').agg({'count': 'sum'})
    usptbycounty = usptbycounty.reset_index()
    
    # merge the PT county counts with the rest of the population density data and mark join misses as 0
    ptdat = usdat.merge(usptbycounty, how='left', on='GEOID').rename(columns={'count': 'npts'})
    ptdat['npts'] = ptdat['npts'].apply(lambda x: x if pd.notnull(x) else np.NaN)
    
    # number of pts normalized by population (pts per thousand people)
    ptdat['ptp1k'] = [1000 * x.npts / x.population if (pd.notnull(x.npts) and pd.notnull(x.population)) else np.NaN 
                      for ii, x in ptdat.iterrows()]
    
    stateptdat = ptdat.loc[:,['STATEFP', 'npts']].groupby('STATEFP').sum()
    stateptdat.rename(columns={'npts': 'statepts'}, inplace=True)

    ptdat = ptdat.merge(stateptdat.reset_index(), on='STATEFP')
    return ptdat


# for drawing discrete choropleths only
def draw_choropleth2(mapdat, column, cols, fname=None, isbivariate=False, xlabel="x", ylabel="y"):
    cmap = LinearSegmentedColormap.from_list('mymap', cols)
    cmap_discrete = cols
    
    plt.subplots(figsize=(16, 8))
    ax = plt.subplot2grid((8,16), (0,0), rowspan=8, colspan=14)
    
    mapdat.plot(ax=ax, column=column, cmap=cmap)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if not isbivariate:
        nquantiles = len(cols)
        rowspan = 2 if nquantiles<=5 else 4
        ax2 = plt.subplot2grid((8,16), (4,14),rowspan=rowspan)
        vals, labs = get_quantile_labels(nquantiles)
        dat = pd.DataFrame({'vals': vals[1:]})
        sns.heatmap(dat, square=True, cbar=False, cmap=cmap_discrete, xticklabels=False, yticklabels=labs, ax=ax2)
        ax2.invert_yaxis()
        ax2.tick_params(length=0)
        ax2.yaxis.tick_right()
        plt.yticks(rotation='horizontal')
    elif len(cols) == 9:
        ax2 = plt.subplot2grid((8,16), (4,14),rowspan=3, colspan=3)
        dat = np.arange(9).reshape(3,3)
        sns.heatmap(dat, square=True, cbar=False, cmap=cmap_discrete, xticklabels=False, yticklabels=False, ax=ax2)
        ax2.invert_yaxis()
        ax2.tick_params(length=0)
        ax2.set_xlabel(xlabel + ' →')
        ax2.set_ylabel(ylabel + ' →')
    if fname:
        plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.show()
    