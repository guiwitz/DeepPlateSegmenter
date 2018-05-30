import os, re
import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def recover_dataframes(datafolder):
    dataframes = [f for f in os.listdir(datafolder) if re.search('.*.csv', f)]
    
    allframes = []
    for x in dataframes:
        allframes.append(pd.read_csv(datafolder+'/'+x))
    fluoframe = pd.concat(allframes)
    return fluoframe

def fit_dataset(fluoframe):
    
    well_indices = fluoframe.well_name.unique()
    
    fit_results = {'amp1': [], 'mean1': [], 'sigma1': [],
               'amp2': [], 'mean2': [], 'sigma2': [],
              'bins':[], 'hist':[], 'well':[]}

    for i in range(len(well_indices)):
        fluodata = fluoframe[(fluoframe.well_name==well_indices[i])&(fluoframe.probability>0.8)&
                        (fluoframe.eccentricity>0.8)].box3_fluo

        if len(fluodata)<10:
            for k in fit_results.keys():
                fit_results[k].append(np.nan)
            continue
        res, valbins, binmean = fit_fluo_distr(fluodata)
        if len(res)==0:
            for k in fit_results.keys():
                fit_results[k].append(np.nan)
            continue

        binsize = binmean[1]-binmean[0]

        fit_results['bins'].append(binmean)
        fit_results['hist'].append(valbins)
        fit_results['well'].append(well_indices[i])

        if len(res.x)==3:
            fit_results['amp1'].append(res.x[0])
            fit_results['mean1'].append(res.x[1])
            fit_results['sigma1'].append(res.x[2])
            fit_results['amp2'].append(np.nan)
            fit_results['mean2'].append(np.nan)
            fit_results['sigma2'].append(np.nan)

        else:

            if (res.x[2]<2.5)&(res.x[3]>2.5):
                fit_results['amp2'].append(res.x[0])
                fit_results['mean2'].append(res.x[2])
                fit_results['sigma2'].append(res.x[4])

                fit_results['amp1'].append(res.x[1])
                fit_results['mean1'].append(res.x[3])
                fit_results['sigma1'].append(res.x[5])
            else:
                fit_results['amp1'].append(res.x[0])
                fit_results['mean1'].append(res.x[2])
                fit_results['sigma1'].append(res.x[4])

                fit_results['amp2'].append(res.x[1])
                fit_results['mean2'].append(res.x[3])
                fit_results['sigma2'].append(res.x[5])
    return fit_results

def plot_fits(fit_results, folder_to_save):
    pos_seq= np.arange(len(fit_results['amp1']))
    partitioned = [pos_seq[i:i+9] for i  in range(0, len(pos_seq), 9)]

    for s in partitioned:
        fig,ax = plt.subplots(figsize=(10,10))
        for ind, i in enumerate(s):
            plt.subplot(3,3,ind+1)
            plt.title(fit_results['well'][i])
            plt.plot(fit_results['bins'][i],fit_results['hist'][i],'o-')    
            plt.plot(fit_results['bins'][i],fun_double_gauss(fit_results['bins'][i],fit_results['amp1'][i],fit_results['amp2'][i],
                                             fit_results['mean1'][i],fit_results['mean2'][i],
                                             fit_results['sigma1'][i],fit_results['sigma2'][i]),'b')

            plt.plot(fit_results['bins'][i],fun_single_gauss(fit_results['bins'][i],fit_results['amp1'][i],fit_results['mean1'][i],fit_results['sigma1'][i]),'ro-',markersize = 3)
            plt.plot(fit_results['bins'][i],fun_single_gauss(fit_results['bins'][i],fit_results['amp2'][i],fit_results['mean2'][i],fit_results['sigma2'][i]),'c')
            #plt.plot(fine_x,post.fun_single_gauss(fit_results['bins'][i],fit_select['amp'][i],fit_select['mean'][i],fit_select['sigma'][i]),'ro',markersize = 3)
        plt.show() 
        fig.savefig(folder_to_save+'/fit'+str(i)+'.pdf')

def gauss_double_fit(p, *args):
    x,data = args[0], args[1]
    nll = np.sum((fun_double_gauss(x,p[0],p[1],p[2],p[3],p[4],p[5])-data)**2)
    return nll

def gauss_single_fit(p, *args):
    x,data = args[0], args[1]
    nll = np.sum((fun_single_gauss(x,p[0],p[1],p[2])-data)**2)
    return nll

def gauss_var_fit(p, *args):
    x,data = args[0], args[1]
    nll = np.sum((fun_single_gauss(x,p[0],p[1],p[2])-data)**2)
    return nll

def gauss_double_fitfix(p, *args):
    x,data,amp,loc = args[0], args[1], args[2], args[3]
    nll = np.sum((fun_double_gauss(x,amp,p[0],loc,p[1],p[2],p[3])-data)**2)
    return nll

def fun_double_gauss(x, A0,A1, x0, x1,sigma0, sigma1):
    return A0*np.exp(-((x-x0)**2)/(2*sigma0**2)) + A1*np.exp(-((x-x1)**2)/(2*sigma1**2))

def fun_single_gauss(x, A0, x0, sigma):
    return A0*np.exp(-((x-x0)**2)/(2*sigma**2))

def fit_fluo_distr(fluodata):
    
    #find data boundaries
    fluodata = fluodata-100
    fluodata[fluodata<=0]=1
    fluodata= np.log(fluodata)
    valbins, binmean = np.histogram(fluodata, bins=20)
    maxloc = binmean[np.argmax(valbins)]
    populated_bin = binmean[0:-1][valbins>10]
    if populated_bin.shape[0]==0:
        return np.array([]), np.array([]), np.array([])
    
    norm_fact = populated_bin.max()
    fluodata = fluodata/populated_bin.max()
    valbins, binmean = np.histogram(fluodata, bins=20)
    maxloc = binmean[np.argmax(valbins)]
    populated_bin = binmean[0:-1][valbins>10]

    if len(populated_bin)==1:
        varinit = binmean[1]-binmean[0]
    else:
        varinit = np.sqrt(fluodata[(fluodata>populated_bin.min())&(fluodata<populated_bin.max())].var())


    
    #calculate new bins
    bins=np.arange(populated_bin.min()-2*varinit,populated_bin.max()+2*varinit,(4*varinit+populated_bin.max()-populated_bin.min())/50)
    valbins, binmean = np.histogram(fluodata, bins=bins)
    binmean= np.array([0.5*(binmean[x]+binmean[x+1]) for x in range(len(binmean)-1)])
    maxamp = np.max(valbins)
    maxloc = binmean[np.argmax(valbins)]
    
    #fit only the region around the larger peak
    subdatax = binmean[np.argmax(valbins)-5:np.argmax(valbins)+6]
    subdatay = valbins[np.argmax(valbins)-5:np.argmax(valbins)+6]
    
    bounds = ((maxloc-0.1,maxloc+0.1),(0.01,10))
    additional = (subdatax, subdatay)
    initvar = subdatax[1]-subdatax[0]
    
    res1 = scipy.optimize.minimize(fun=gauss_var_fit, args=additional, 
                                              x0=np.array([maxamp, maxloc,2*initvar]),method='BFGS')
    
    #fit the whole distribution with a double gaussian. One of them is based on the parameters obtained in the 
    #subregion which should not vary much. The other is a second gaussian either to the left or to the right.
    additional = (binmean, valbins)
        
    datarange = binmean[-1]-binmean[0]
    minx = binmean[0]+0.33*(binmean[-1]-binmean[0])
    maxx = binmean[0]+0.66*(binmean[-1]-binmean[0])
    bounds = ((res1.x[0]-10,res1.x[0]+10),(0,1*maxamp),(res1.x[1]-0.5,res1.x[1]+0.5),(0,res1.x[1]-0.1*datarange),(res1.x[2]-0.01,res1.x[2]+0.01),(res1.x[2]-0.02,res1.x[2]+0.02))
    res2 = scipy.optimize.minimize(fun=gauss_double_fit, args=additional, 
                                          x0=np.array([res1.x[0],0.15*maxamp,res1.x[1],minx,res1.x[2],res1.x[2]]),
                                          method='L-BFGS-B',bounds = bounds)
    
    
    minx = binmean[0]+0.66*(binmean[-1]-binmean[0])
    maxx = 1

    bounds = ((res1.x[0]-10,res1.x[0]+10),(0,1*maxamp),(res1.x[1]-0.5,res1.x[1]+0.5),(res1.x[1]+0.1*datarange,binmean[-1]),(res1.x[2]-0.01,res1.x[2]+0.01),(res1.x[2]-0.02,res1.x[2]+0.02))
    res3 = scipy.optimize.minimize(fun=gauss_double_fit, args=additional, 
                                          x0=np.array([res1.x[0],0.15*maxamp,res1.x[1],minx,res1.x[2],res1.x[2]]),
                                          method='L-BFGS-B',bounds = bounds)
    
    #take the best fitting solution.
    if res2.fun<res3.fun:
        res = res2
    else:
        res = res3
        
    res.x[2:6] = norm_fact*res.x[2:6]
    
    binmean = norm_fact*binmean
    return res, valbins,binmean