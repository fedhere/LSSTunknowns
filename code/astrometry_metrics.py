import numpy as np
import pandas as pd  
from builtins import zip  
from scipy.stats import norm, uniform  
from sklearn.neighbors import KernelDensity   
from lsst.sims.maf.metrics import BaseMetric  
from lsst.sims.maf.utils import m52snr 
from opsimUtils import *  
from astropy.io import fits   
from sklearn.utils import shuffle
import sklearn.mixture as GMM  
from lsst.sims.maf.utils.mafUtils import radec2pix

def position_selection(R,z):
            # costants
            ab = 1 #kpc
            ah = 1.9 #kpc
            alphab = 1.8
            alphah = 1.63
            betah =2.17
            rb = 1.9 #Kpc
            qb = 0.6
            qh = 0.8
            z0 = 0.3 #kpc
            z1 = 1 #kpc
            Rd = 3.2 #kpc
            rho0b = 0.3*10**(-9) #Mskpc^-3
            rho0h = 0.27*10**(-9) #Mskpc^-3
            sigma = 300*10**(-6) #Mskpc^-2
            alpha1 = 0.5
            # parametes
            mb = np.sqrt(R**2+z**2/qb**2)
            mh = np.sqrt(R**2+z**2/qh**2)
            alpha0 =1-alpha1
            
            rhoh = rho0h*(mh/ah)**alphah*(1+mh/ah)**(alphah-betah)
            rhob = rho0b*(mb/ab)**(-alphab)*np.exp(-mb/rb)
            rhod = sigma*np.exp(R/Rd)*(alpha0/(2*z0)*np.exp(-np.absolute(z)/z0)+alpha1/(2*z1)*np.exp(-np.absolute(z)/z1))
            densities = np.array([rhoh,rhob,rhod])/np.sum(np.array([rhoh,rhob,rhod]))
            mode = np.array(['H','B','D'])
            idx = np.where(densities == np.amax(densities))
            return mode[idx]
        
def DF(V_matrix,mode,R,z):                
            if mode == 'H':
                v = np.sqrt(V_matrix[0,:]**2+V_matrix[1,:]**2+V_matrix[2,:]**2)
                vesc = 575 #km/s
                vsun =187.5 #km/s
                N=1.003
                P=4*N/vsun/np.sqrt(np.pi)*(v/vesc)**2*np.exp(-v**2/vsun)
            if mode == 'B': 
                v = np.sqrt(V_matrix[0,:]**2+V_matrix[1,:]**2+V_matrix[2,:]**2)
                disp = 140 #km/s
                P = np.exp(-v**2/2/disp**2)/np.sqrt(np.pi)/disp
            if mode == 'D':
                k= 0.25
                q =0.45
                Rd = 3.2 #kpc
                sigmaz0= 19 #km/s
                sigmar0= 33.5 #km/s
                beta = 0.33
                L0 = 10 #km/s
                sigma = 300*10**(-6) #Ms/kpc**2
            
                # parameters
                v = V_matrix[2,:]
                Jz = v**2/2 + v**2/2*np.log(R**2+z**2/0.8**2) 
                Ar = k * Jz /np.exp(2*q*(120 - R)/Rd)/sigmar0
                Az =  k * Jz /np.exp(2*q*(120 - R)/Rd)/sigmaz0                
                P = v**2/Jz*sigma*(1+np.tanh(R*v/L0))*np.exp(-k*Jz/(sigmar0*np.exp(2*q*(120 - R)/Rd))**2)/(np.pi*k*sigmar0*np.exp(2*q*(120 - R)/Rd))
            return P
        
def simulate_pm(nexp=1000,M_min=15,M_max=25, prob_type='uniform',U=np.arange(-100,100,25),V=np.arange(-100,100,25),W=np.arange(-100,100,25)):
    np.random.seed(5000) 
    names=['d','MODE','PM', 'PM_out','mag'] 
    variables = {k:[] for k in names}#np.zeros(self.nexp, dtype=list(zip(names, [float]*len(names)))) 
    g_dist = pd.read_csv('absMag_g-dist.csv', usecols=['mag_g', 'frequency'], index_col=False)
    age_dist = pd.read_csv('time-dist.csv', usecols=['age', 'frequency'], index_col=False)
    fout = open('simulation_pm.csv', 'w+')
    fout.write('MAG'+','+'MODE'+','+'d'+','+'PM'+','+'PM_out'+'\n')
    n=0
    while n < nexp:
        g_abs = np.random.choice(g_dist['mag_g'],p=g_dist['frequency']) # magAB 
        age =  np.random.choice(age_dist['age'],p=age_dist['frequency']) # magAB 
        R = np.random.uniform(0, 120) # kpc 
        z = np.random.uniform(-120*0.8, 120*0.8) # kpc 
        d = np.sqrt(R**2+z**2)*10**(3)
        M = g_abs +5*np.log10(d)-5 
        if (M>M_min) and (M< M_max):  
            n+=1
            print(n)
            mode = position_selection(R,z)
            V_matrix=np.vstack((U,V,W))
            velmat = np.empty_like(V_matrix[0,:])
            veloutmat = np.empty_like(V_matrix[0,:])
            Pv=DF(V_matrix,mode,R,z)  
            vel_idx=np.random.choice(np.arange(0,len(V_matrix[0,:]),1)[np.isfinite(Pv)],p=Pv[np.isfinite(Pv)]/np.sum(Pv[np.isfinite(Pv)]), size = 3)
            if prob_type=='uniform':
                vel_out= uniform(-1000,1000) 
                veloutmat = vel_out.rvs(size=3)
            else:
                vel_out = pd.read_csv(prob_type)
                veloutmat = np.random.choice(vel_out['vel'],p=vel_out['fraction']/np.sum(vel_out['fraction']),size=3)
                    
            velmat = V_matrix[0,vel_idx]
            vT = velmat[2]
            vfT = veloutmat[2]
            direction =np.random.choice((-1, 1))
            mu = direction*vT/4.75/d 
            muf = direction*vfT/4.75/d 
            fout.write('{}'.format(M)+','+'{}'.format(mode)+','+'{}'.format(d)+','+'{}'.format(mu)+','+'{}'.format(muf)+' \n')    
        else:
            continue
    return print('simulated pm of {} objects'.format(nexp))   
    
    
class get_cols(BaseMetric):
    def __init__(self, metricName='get_cols', cols=[], **kwargs): 
            self.cols=cols  
            super(get_col, self).__init__(col=cols, metricName=metricName, **kwargs) 
    def run():
        val= {k:dataSlice[k] for k in self.cols}
        return val
        
class TransienPM(BaseMetric): 
     #    Generate a population of transient objects and see what is its proper motion  , 
    def __init__(self, metricName='TransienPM', nexp=100, Mmin=10, Mmax=28, U=np.arange(-400,400,25), f='g', snr_lim=5, 
                  V=np.arange(-600,200,25),W=np.arange(-400,400,25),m5Col='fiveSigmaDepth',  
                  mjdCol='observationStartMJD',filterCol='filter',seeingCol='seeingFwhmGeom', surveyduration=10, **kwargs): 
            self.mjdCol = mjdCol 
            self.seeingCol= seeingCol 
            self.m5Col = m5Col 
            self.filterCol = filterCol 
            self.nexp = nexp 
            self.snr_lim = snr_lim 
            self.f = f 
            self.U=U 
            self.V=V 
            self.W=W 
            self.Mmin= Mmin 
            self.Mmax= Mmax 
            self.surveyduration = surveyduration  
            sim = pd.read_csv('simulation_pm.csv', usecols=['MAG','MODE','d','PM','PM_out'])
            self.simobj = sim
            super(TransienPM, self).__init__(col=[self.mjdCol, self.m5Col,self.seeingCol, self.filterCol], 
                                                       units='Fraction Detected', 
                                                       metricName=metricName, **kwargs) 
      
         # typical velocity distribution from litterature (Binney et Tremain- Galactic Dynamics) 
      
    def lightCurve(self, t, t0, peak, duration, slope): 
     #      A simple top-hat light curve., 
     #         
     #        Parameters , 
     #        ---------- , 
     #        t : array , 
     #            Times to generate lightcurve points (mjd) , 
     #        t0 : float , 
     #            Initial time (mjd) , 
     #        m_r_0 : float , 
     #            initial r-band brightness (mags) , 
             lightcurve = np.zeros(np.size(t), dtype=float) + 99. 
     # Select only times in the lightcurve duration window , 
             good = np.where( (t >= t0) & (t <= t0+duration) ) 
             lightcurve[good] = peak + slope*(t[good]-t0) 
             return lightcurve 
      
    def run(self,  dataSlice, slicePoint=None): 
            np.random.seed(5000)
            obs = np.where((dataSlice['filter']==self.f) & (dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration)) 
            names=['PM', 'PM_out','snr','timegap'] 
            variables = {k:None for k in names} 
            if len(dataSlice[self.m5Col][np.where(dataSlice['filter'][obs]==self.f)])>2:     
                tf = dataSlice[self.mjdCol][np.where(dataSlice['filter'][obs]==self.f)] 
                deltaT = np.absolute(tf - shuffle(tf)) # dd                                                                           
                seeing=dataSlice[self.seeingCol] # arcsec 
                 #g_abs = np.random.choice(g_dist['mag_g'],p=g_dist['frequency']) # magAB  
                    #d = self.simobj['d']#np.random.uniform(50, 1.2*10**5) # pc    
                N = 0
                for i, (dT, see, ra,fiveSlim) in enumerate(zip(deltaT,seeing,dataSlice['fieldRA'][obs],dataSlice[self.m5Col][obs])):

                    d = self.simobj['d']
                    M = self.simobj['MAG']
                    snr = m52snr(M,fiveSlim) # arcsec) 
                    mode = self.simobj['MODE']
                    mu = self.simobj['PM']
                    muf = self.simobj['PM_out']
                    arc = np.absolute(muf)*dT/365

                    good = np.where((arc > 0.05*see) & (snr>self.snr_lim))[0]
                    if np.size(good)>0:
                        variables['timegap']= dT 
                        variables['PM_out']=muf[good]
                        variables['PM']=mu[good] 
                        variables['snr']= snr[good]                     
                        objRate = 0.7 # how many go off per day , 
                # decide how many objects we want to generate , 
                        nObj=len(variables['PM']) 
                        N +=nObj
                        m0s = np.array(M[good])
                        t = dataSlice[self.mjdCol][obs] - dataSlice[self.mjdCol].min() 
                        detected = 0 
                 # Loop though each generated transient and decide if it was detected , 
                 # This could be a more complicated piece of code, for example demanding  , 
                 # A color measurement in a night. , 

                        for i,t0 in enumerate(np.random.uniform(0,self.surveyduration,nObj)): 
                            duration =dT
                            slope = np.random.uniform(-3,3) 
                            lc = self.lightCurve(t, t0, m0s[i],duration, slope) 
                            good = m52snr(lc,dataSlice[self.m5Col][obs])> self.snr_lim 
                            detectTest = dataSlice[self.m5Col][obs] - lc 
                            if detectTest.max() > 0 and len(good)>2: 
                                 detected += 1 
                 # Return the fraction of transients detected , 
                if float(N) == 0:
                    A = np.inf 
                else: 
                    A=float(N) 
                    res = float(np.sum(detected))/A            
                    #print('detected fraction:{}'.format(res)) 
                    return res
            #else:
                #print('no detected pm for transients')
                #continue
          
          


class confusionmetric(BaseMetric): 
        def __init__(self, filename = 'data.csv', colsname=['RA', 'DEC','g','g-r','Hg','PM', 'deltaX'], snr_lim=5,mode=None, MagIterLim=[0,1,1], surveyduration=10,
                      metricName='confusionmetric',m5Col='fiveSigmaDepth',  
                      mjdCol='observationStartMJD',filterCol='filter', seeingCol='seeingFwhmGeom',dataout=True,**kwargs): 
            self.mjdCol = mjdCol 
            self.m5Col = m5Col 
            self.seeingCol = seeingCol 
            self.filterCol = filterCol 
            self.colsname = colsname 
            self.snr_lim = snr_lim 
            self.filename = filename  
            self.dataout = dataout 
            self.mode = mode 
            self.MagIterLim = MagIterLim 
            self.surveyduration = surveyduration 
             # to have as output all the simulated observed data set dataout=True, otherwise the relative error for  
             # each helpix is estimated 
            if self.dataout: 
                super(confusionmetric, self).__init__(col=[self.mjdCol,self.filterCol, self.m5Col,self.seeingCol, 'night' ],metricDtype='object', units='', metricName=metricName, 
                                                      **kwargs) 
            else: 
                super(confusionmetric, self).__init__(col=[self.mjdCol,self.filterCol,  self.m5Col,self.seeingCol, 'night' ], 
                                                            units='', metricDtype='float',metricName=metricName, 
                                                             **kwargs) 
        def readfile(self, filename='', colsname=['']): 
            if 'csv' in filename: 
                #print('reading {}'.format(filename)) 
                data = pd.read_csv(filename, header=0, names = colsname ) 
            elif 'fits' in filename: 
                 #print('reading {}'.format(filename)) 
                hdul = fits.open(filename) 
                data = hdul[1].data 
            elif ['txt', 'dat'] in filename: 
                #print('reading {}'.format(filename)) 
                data = {k:[] for k in colsname} 
                f = open(filename) 
                righe = f.readlines() 
                for line in righe: 
                    line.split() 
                    for i, k in enumerate(colsname): 
                        data[k].append(float(line[i]))                 
            elif 'json'in filename: 
                print('not implemented to read .json extention') 
            return data 
        def run(self, dataSlice, slicePoint=None): 
              
            colsname=['RA', 'DEC','g','g-r','Hg','PM','deltaX'] 
            data = self.readfile(self.filename, self.colsname) 
            obs = np.where(dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration) 
            m5 = dataSlice[self.m5Col][obs] 
            seeing= dataSlice[self.seeingCol][obs] 
            filters = dataSlice['filter'][obs] 
            g = np.array(data[self.colsname[2]]) 
            Hg = np.array(data[self.colsname[4]]) 
            gr = np.array(data[self.colsname[3]]) 
            pm = np.array(data[self.colsname[5]])*1e-3 
            deltamag= np.arange(self.MagIterLim[0],self.MagIterLim[1],self.MagIterLim[2]) 
            out = {} 
            fwhm = {} 
            mag_lim = np.median(m5[np.where(filters==self.colsname[2])]) 
            for dm in deltamag: 
                if self.mode == 'distance': 
                    pmnew= pm/(10**(dm/5)) 
                elif self.mode == 'density': 
                    pmnew= pm 
                else: 
                    print('##### ERROR: the metric is not implemented for this mode.') 
                s=np.median(seeing) 
                sigmafw=0.67*s 
                coeff= np.log(10)*5 
                deltaX= pmnew*self.surveyduration 
                good=np.empty(len(Hg)) 
                for i, (h, p) in enumerate(zip(Hg,pmnew)): 
                    mag= g[i]+dm 
                    snr = m52snr(mag,mag_lim) 
                    if snr>self.snr_lim and deltaX[i] > 0.05*s: 
                        sigmaHg=np.sqrt(snr**(-2)*(1+(sigmafw*coeff/p/self.surveyduration)**2)) 
                        good_Hg = np.size(np.where((Hg>h-sigmaHg) &(Hg< h+sigmaHg)))/np.size(Hg) 
                        good_gr = np.size(np.where((gr>gr[i]-snr**(-1)) &(gr<gr[i]+snr**(-1))))/np.size(Hg) 
                        if good_Hg==good_gr: 
                            good[i] = good_Hg 
                        else: 
                            good[i] = max([good_Hg,good_gr]) 
                    else: 
                        good[i] = np.nan 
                out[dm] = {'num':good, 'Hg': Hg,'g-r': gr}  
            if self.dataout: 
                return out  
            else: 
                if np.isnan(out[dm]['num']).all(): 
                    res= np.nan 
                else: 
                    N=62 
                    d= np.column_stack([gr,Hg]) 
                    models = GMM.GaussianMixture(2, covariance_type='full', random_state=0,init_params='kmeans').fit(d) 
                    labels=models.predict(d) 
                    DM = np.array([out[dm]['num'] for dm in out.keys()]) 
                    LM = np.array([labels[np.where(np.isfinite(out[dm]['num']))] for dm in out.keys()]) 
                    alpha = N*len(LM[np.where(LM==0)])/len(DM[np.isfinite(DM)])
                    res = (1-np.median(DM[np.isfinite(DM)]))/alpha 
                    #print('Confusion Index={}'.format(res)) 
                    return res 

                
                
                
                
class LSPMmetric(BaseMetric): 
        def __init__(self, metricName='LSPMmetric', f='g',surveyduration=10,snr_lim=5,mag_lim=[17,25],percentiles=[2.5,97.5,50], 
                    U=np.arange(-100,100,25),V=np.arange(-100,100,25),W=np.arange(-100,100,25),unusual='uniform',m5Col='fiveSigmaDepth',  
                    mjdCol='observationStartMJD',filterCol='filter', seeingCol='seeingFwhmGeom', nexp= 1,dataout=False, 
                    **kwargs): 
            self.mjdCol = mjdCol 
            self.m5Col = m5Col 
            self.seeingCol = seeingCol 
            self.filterCol= filterCol 
            self.surveyduration =surveyduration 
            self.percentiles = percentiles
            self.snr_lim = snr_lim 
            self.mag_lim = mag_lim
            self.f = f 
            self.nexp = nexp 
            self.U=U 
            self.V=V 
            self.W=W 
            self.unusual=unusual
            self.dataout = dataout 
             # to have as output all the simulated observed data set dataout=True, otherwise the relative error for  
             # each helpix is estimated 
            sim = pd.read_csv('simulation_pm.csv', usecols=['MAG','MODE','d','PM','PM_out'])
            self.simobj = sim
            if self.dataout: 
                super(LSPMmetric, self).__init__(col=[self.mjdCol,self.filterCol, self.m5Col,self.seeingCol, 'night' ],metricDtype='object', units='', metricName=metricName, 
                                                      **kwargs) 
            else: 
                super(LSPMmetric, self).__init__(col=[self.mjdCol,self.filterCol,  self.m5Col,self.seeingCol, 'night'], 
                                                            units='Proper Motion relative error', metricName=metricName, 
                                                             **kwargs) 
          
            np.seterr(over='ignore',invalid='ignore')
         # typical velocity distribution from litterature (Binney et Tremain- Galactic Dynamics)
            
        def position_selection(self,R,z):
            # costants
            ab = 1 #kpc
            ah = 1.9 #kpc
            alphab = 1.8
            alphah = 1.63
            betah =2.17
            rb = 1.9 #Kpc
            qb = 0.6
            qh = 0.8
            z0 = 0.3 #kpc
            z1 = 1 #kpc
            Rd = 3.2 #kpc
            rho0b = 0.3*10**(-9) #Mskpc^-3
            rho0h = 0.27*10**(-9) #Mskpc^-3
            sigma = 300*10**(-6) #Mskpc^-2
            alpha1 = 0.5
            # parametes
            mb = np.sqrt(R**2+z**2/qb**2)
            mh = np.sqrt(R**2+z**2/qh**2)
            alpha0 =1-alpha1
            
            rhoh = rho0h*(mh/ah)**alphah*(1+mh/ah)**(alphah-betah)
            rhob = rho0b*(mb/ab)**(-alphab)*np.exp(-mb/rb)
            rhod = sigma*np.exp(R/Rd)*(alpha0/(2*z0)*np.exp(-np.absolute(z)/z0)+alpha1/(2*z1)*np.exp(-np.absolute(z)/z1))
            densities = np.array([rhoh,rhob,rhod])/np.sum(np.array([rhoh,rhob,rhod]))
            mode = np.array(['H','B','D'])
            idx = np.where(densities == np.amax(densities))
            return mode[idx]
        
        def DF(self,V_matrix,mode,R,z):                
            if mode == 'H':
                v = np.sqrt(V_matrix[0,:]**2+V_matrix[1,:]**2+V_matrix[2,:]**2)
                vesc = 575 #km/s
                vsun =187.5 #km/s
                N=1.003
                P=4*N/vsun/np.sqrt(np.pi)*(v/vesc)**2*np.exp(-v**2/vsun)
            if mode == 'B': 
                v = np.sqrt(V_matrix[0,:]**2+V_matrix[1,:]**2+V_matrix[2,:]**2)
                disp = 140 #km/s
                P = np.exp(-v**2/2/disp**2)/np.sqrt(np.pi)/disp
            if mode == 'D':
                k= 0.25
                q =0.45
                Rd = 3.2 #kpc
                sigmaz0= 19 #km/s
                sigmar0= 33.5 #km/s
                beta = 0.33
                L0 = 10 #km/s
                sigma = 300*10**(-6) #Ms/kpc**2
            
                # parameters
                v = V_matrix[2,:]
                Jz = v**2/2 + v**2/2*np.log(R**2+z**2/0.8**2) 
                Ar = k * Jz /np.exp(2*q*(120 - R)/Rd)/sigmar0
                Az =  k * Jz /np.exp(2*q*(120 - R)/Rd)/sigmaz0                
                P = v**2/Jz*sigma*(1+np.tanh(R*v/L0))*np.exp(-k*Jz/(sigmar0*np.exp(2*q*(120 - R)/Rd))**2)/(np.pi*k*sigmar0*np.exp(2*q*(120 - R)/Rd))
                
            return P
        

          
        def Likelihood(self, variables, ty= ''): 
            if ty == 'usual': 
                pmname = 'PM'
                pmnamex = 'PMx' 
                pmnamey = 'PMy'
            elif ty == 'unusual': 
                pmname = 'PM_out' 
                pmnamex = 'PMx_out' 
                pmnamey = 'PMy_out' 
            else: 
                 print('you need to define the proper motion object type [usual or unusual]') 
            
            # pm dist conditioned to the mag and the dt
            B = np.where(variables['MODE']=='B')[0]
            H = np.where(variables['MODE']=='H')[0]
            D = np.where(variables['MODE']=='D')[0]
            
            if len(B)>0:
                Lb, binsb = np.histogram(variables[pmname][B],bins=np.linspace(np.amin(variables[pmname][B]),np.amax(variables[pmname][B]),np.size(variables['MODE'])+1))
                pi_b = len(B)/len(variables['MODE'])
            else:
                Lb = np.zeros(np.size(variables['MODE']))
                pi_b=0
            if len(D)>0:
                Ld, binsd = np.histogram(variables[pmname][D],bins=np.linspace(np.amin(variables[pmname][D]),np.amax(variables[pmname][D]),np.size(variables['MODE'])+1))
                pi_d = len(D)/len(variables['MODE'])
            else:
                Ld = np.zeros(np.size(variables['MODE']))
                pi_d=0
            if len(H)>0:
                Lh, binsh = np.histogram(variables[pmname][H],bins=np.linspace(np.amin(variables[pmname][H]),np.amax(variables[pmname][H]),np.size(variables['MODE'])+1))
                pi_h = len(H)/len(variables['MODE'])
            else:
                Lh = np.zeros(np.size(variables['MODE']))
                pi_h=0
            
            Lk = np.array([pi_b*Lb,pi_d*Ld,pi_h*Lh])
            Pm = np.exp(-0.5*variables['snr']**2*((variables[pmnamex]*variables['d']*variables['t']-variables['x'])**2+(variables[pmnamey]*variables['d']*variables['t']-variables['y'])**2)/(variables['FWHM']**2*0.67**2*variables['d']**2*variables['t']**2))*variables['snr']/(np.sqrt(2*np.pi)*variables['FWHM']*0.67)
            L = np.sum(Lk, axis=0)/len(Lk) *  Pm/np.sum(Pm) 
            return L 
          
                           
        def run(self, dataSlice, slicePoint=None): 
            np.random.seed(2500) 
            names=['d','x','y','MODE','PM', 'PM_out','PMx', 'PMx_out','PMy', 'PMy_out','FWHM','snr','mag', 't'] 
            variables = {k:[] for k in names}#np.zeros(self.nexp, dtype=list(zip(names, [float]*len(names)))) 
            #g_dist = pd.read_csv('absMag_g-dist.csv', usecols=['mag_g', 'frequency'], index_col=False)
            #age_dist = pd.read_csv('time-dist.csv', usecols=['age', 'frequency'], index_col=False)
            obs = np.where((dataSlice['filter']==self.f) & (dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration)) 
            if len(dataSlice[self.m5Col][obs])>2: 
                tf = dataSlice[self.mjdCol][obs] 
                seeing = dataSlice[self.seeingCol][obs] # arcsec                             
                deltaT = np.absolute(tf - shuffle(tf)) # dd  
                for i, (dT, see, ra,fiveSlim) in enumerate(zip(deltaT,seeing,dataSlice['fieldRA'][obs],dataSlice[self.m5Col][obs])):
                        d = self.simobj['d']
                        M = self.simobj['MAG']
                        snr = m52snr(M,fiveSlim) # arcsec 
                        mode = self.simobj['MODE']
                        mu = self.simobj['PM']
                        muf = self.simobj['PM_out']
                        arc = np.absolute(muf)*dT 
                        good = np.where((arc > 0.05*see) & (snr>self.snr_lim))[0]
                        if np.size(good)>0:
                            variables['x'].append(np.ones(len(good))*ra)
                            variables['y'].append(np.ones(len(good))*dataSlice['fieldDec'][obs][i])
                            variables['d'].append(d[good])
                            variables['MODE'].append(mode[good])
                            variables['PM_out'].append(muf[good]) 
                            variables['PMx_out'].append(muf[good]*np.cos(dataSlice['fieldDec'][obs][i]))
                            variables['PMy_out'].append(muf[good]*np.sin(dataSlice['fieldDec'][obs][i]))
                            variables['PM'].append(mu[good]) 
                            variables['PMx'].append(mu[good]*np.cos(dataSlice['fieldDec'][obs][i]))
                            variables['PMy'].append(mu[good]*np.sin(dataSlice['fieldDec'][obs][i]))
                            variables['t'].append(np.ones(len(good))*dT) 
                            variables['FWHM'].append(np.ones(len(good))*see) 
                            variables['snr'].append(np.ones(len(good))*fiveSlim) 
                            variables['mag'].append(M[good])    
                            variables = {k:np.concatenate(variables[k]) for k in variables.keys()}
                            
                            
                            p_min,p_max,p_mean = self.percentiles[0],self.percentiles[1],self.percentiles[2]
                            mu_min,mu_max, mu_mean = np.percentile(variables['PM'],[p_min,p_max,p_mean])
                            OK = np.isfinite(variables['PM_out'])
                            pm = variables['PM_out'][OK]
                            muf_index = np.where((pm < mu_min)| (pm >mu_max))
                            unusual = np.size(muf_index)
                            res = unusual/len(variables['PM_out'])
                            if self.dataout: 
                                fieldRA = np.mean(dataSlice['fieldRA']) 
                                fieldDec = np.mean(dataSlice['fieldDec'])
                                like_PM = self.Likelihood(variables, ty='usual') 
                                like_PMout = self.Likelihood(variables, ty='unusual')
                                dic = {'detected': res,
                                      'pixID': radec2pix(nside=16, ra=np.radians(fieldRA), dec=np.radians(fieldDec))}
                                      #'L_PM':like_PM,
                                       #'L_PMout': like_PMout,
                                       #'mag':variables['mag'],
                                       #'PM': [min(variables['PM']),max(variables['PM'])], 
                                       #'PM_OUT': [min(variables['PM_out']),max(variables['PM_out'])]}                                 
                                return dic 
                            else: 
                                return res 
                  