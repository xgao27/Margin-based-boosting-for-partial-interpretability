# some useful functions
import pandas as pd
import numpy as np
import math
import time

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,ShuffleSplit

from sklearn import datasets
from sklearn.model_selection import train_test_split,KFold
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# print stuff ######################################################################################
def printTime():
    print("Latest excution of this block:", time.ctime(time.time()))

def printClassifier(f,verbose = 0):
    print("----------------- classifier summary -----------------")
    if verbose>0:
        print ("feature of h_t \t","\t a_t")
        for i in range(len(f['h'])):
            print ("Feature index: ",np.argmax(f['h'][i].feature_importances_),"\t",f['a'][i])
    print ("total number of h:", len(f['h']), " total a_t: ",np.sum(f['a']))
    print ()
    
def printClassifierFeatures(f,print_weights=True):
    features = []
    weights = []
    for i in range(len(f['h'])):
        features.append(np.argmax(f['h'][i].feature_importances_))
        weights.append(f['a'][i])
    print("----------------- classifier summary -----------------")
    print(features)
    if print_weights: print(weights)
        
# get data ######################################################################################          
def loadData(name,label_map=[0,1],label_ind = -1,exclude=[],ratio=0.25,max_row=-1):
    if "/" not in name:
        name = './data/'+name
    df_all = pd.read_csv(name)
    # if zero_one:
    #     y = (df_all.values[:,label_ind]*2-1.0).astype(np.int64)
    # else:
    #     y = df_all.values[:,label_ind]
    
    y = df_all.values[:,label_ind]
    y = np.where(y==label_map[0],-1,y)
    y = np.where(y==label_map[1],1,y)
    
    exclude.append(label_ind)
    X = np.delete(df_all.values, exclude, 1)
    if max_row>0:
        perm = np.random.permutation(X.shape[0])
        X = X[perm][:max_row]
        y = y[perm][:max_row]
    
    print("--------- dataset summary: ---------")
    print(df_all.info())
    print (X.shape, y.shape)
    print (X[:5],y[:5])
    
    return X,y
    
def map_label(y,zero_one):
    if zero_one == 0: # input [-1, 1], want [0,1] 
        new_y = np.where(y==-1,0,y)
    else: # input [0,1], want [-1, 1] 
        new_y = np.where(y==0,-1,y)
    return new_y
    
def generateData(shape,d,eta,noise_type=0,split=True,test_ratio=0.25,verbose=True):
    (m,D) = shape
    X = np.random.randint(0,2,shape).astype(np.float32) # 0, 1 binary data
    y = np.zeros(m)
    w = np.abs(np.random.normal(loc=1.0,scale=1.0,size=D)) 
    w[d:] = 0 # d weights 
    w = w / np.sum(w)
    b = 0.5
    y = np.matmul(X,w)
    y = np.where(y>b,1.0,-1.0) # true linear labels
    noise = np.array([])
    coeff = np.array([])
    degree = 0
    if eta>0:
        noise = np.argwhere(eta > np.random.uniform(0,1,m))[:,0]
  
        if noise_type==1:
            y[noise] = (np.random.randint(0,2,noise.shape[0])-0.5)*2 # random noise
        elif noise_type==2:
            noise = np.random.permutation(m)[:int(np.floor(m*eta))]
            y[noise] = y[noise] * (-1) # white-label noise for eta fraction
        elif noise_type==3:
            degree = 2
            y_sum = 0
            coeff = np.random.normal(loc=0,scale=1.0,size=degree+1) 
            for i in noise:
                x = np.zeros(degree+1)
                for j in range(degree+1):
                    x[j] = np.power((X[i,d+j]-0.5), j )
                y[i] = np.dot(x,coeff)
                y_sum+= y[i]
            y_avg = y_sum / noise.shape[0]
            b = 0
            y[noise] = np.where(y[noise]>b,1.0,-1.0)
            # X[noise,-1] = 0.5
        else:
            y[noise] = y[noise] * (-1) # white-label noise w.p. eta
    if verbose:
        print("generated data of shape: ",shape,"noisy data {0}".format(noise.shape[0]), ", 1 vs -1 :", np.where(y>0)[0].shape[0], np.where(y<0)[0].shape[0] )
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_ratio, random_state=42)  
    if noise_type==3:
        poly = {'coeff':np.copy(coeff),'b':b,'ind':[d+j for j in range(degree+1)]}
        if split:
            return (X_train, X_test, y_train, y_test,w,b,poly)
        else:
            return (X,y,w,b,poly)
    if split:
        return (X_train, X_test, y_train, y_test,w,b)
    else:
        return (X,y,w,b)

# classifier related ######################################################################################
def marginError(X,y,f,theta = 0):
    # calculates P_X[y * f(x) < theta] 
    # if X is training data, this is P_S, if X is testing data this is P_D
    # if theta = 0, this is just error
    pred = weightedVote(f,X)
    m = X.shape[0]
    # D = np.ones(m)*1.0/m
    # error = D.dot(pred*y < theta)
    error = np.sum(pred*y < theta)*1.0/m
    return error
    

def weightedVote(f,X):
    # calculates f(x) = \sum{a_i * h_i(x)} for each x \in X, normalized by \sum{a_i}
    T = len(f['h'])
    m = X.shape[0]
    pred = np.zeros((m,T))
    for t in range(T):
        pred[:,t] = f['a'][t] * f['h'][t].predict(X)
    y = np.sum(pred,axis=1) /np.sum(f['a'])
    return y

def cover(f,X,theta,reverse=False):
    # this should have been written as y_pred, theta, but ok...
    # margin = |f(x)|
    # returns indices of covered data points (where margin > theta), and coverage %
    ind = np.arange( X.shape[0])
    margin = np.abs(weightedVote(f,X))
    
    if reverse:
        ind = ind[margin<theta]
    else:
        ind = ind[margin>=theta]
    return ind, ind.shape[0]*1.0/X.shape[0]
    
# adaBoost ######################################################################################    
def adaBoost(X,y,T,baseLearner = DecisionTreeClassifier,kwargs={'max_depth':1}):
    (m,d) = X.shape
    D_t = np.ones(m)*1.0/m
    h_list = [] # list of h_i
    a_list = [] # list of corresponding weights a_i
    for t in range(T):
        h_t = baseLearner(**kwargs)
        h_t.fit(X,y,sample_weight = D_t)
        
        y_t = h_t.predict(X)
        e_t = D_t.dot(y_t != y) # sample weighted (using D_t) empirical error
        if(e_t < 0.5):
            a_t = 0.5 * np.log((1 - e_t)/(e_t))
            Z_t = 2*np.sqrt(e_t*(1-e_t))
            D_t = D_t * np.exp(- a_t * y * y_t)/Z_t
            
            h_list.append(h_t)
            a_list.append(a_t)
    return {'h':h_list,'a':a_list}

# roBoost ######################################################################################
def positivePart(diff):
    # returns (diff)_+
    return np.where(diff > 0,diff,0)

def G(a,prevR,prevA,yhx,nu,prev_r_tilde, forOPT = True,r_lb = -1.0):
    # used for optimizing a_t, takes a as the opt variable
    def argmax_r (r,new_r,nu):
        return -(r - np.sum(positivePart(r - new_r ))/(nu*new_r.shape[0]))
    # given new a and yhx, get newRho
    # newR, newA = updateRho(prevR,prevA, a, yhx)
    newR = np.copy(prevR + a * yhx) # un-normalized
    newA = prevA + a
    new_r = newR/newA
    if nu>0:
        res = minimize(argmax_r, [0.0], args = (new_r,nu), bounds = [(r_lb,1.0)],tol=1e-6)
        r_OPT = res.x[0]
    else:
        r_OPT = np.min(new_r) # if nu = 0, r_opt is the min margin based on current a's
    if forOPT:
        xi = positivePart(r_OPT - new_r)
        xi_sum = np.sum(xi)
        if nu>0:
            exp_loss = np.sum(np.exp(-newA*(new_r + xi - xi_sum/(nu*new_r.shape[0])- 1*prev_r_tilde)))
        else:
            exp_loss = np.sum(np.exp(-newA*(new_r - prev_r_tilde))) #-r_OPT*newA 
        return exp_loss
    else:
        return r_OPT
    
# def G_(new_r,nu):
#     # a_t already set, just need to figure out what the rho_opt was for this a_t (aka newRho)
#     def argmax_r (r,new_r,nu):
#         return -(r - np.sum(positivePart(r - new_r ))/(nu*new_r.shape[0]))
    
#     res = minimize(argmax_r, [0], args = (new_r,nu), bounds = [(-1.0,1.0)],method='Nelder-Mead', tol=1e-6)
#     r_opt = res.x[0] 
#     return r_opt
    
def roBoost(X,y,T,nu,baseLearner = DecisionTreeClassifier,kwargs={'max_depth':1},verbose=0,r_lb=-1.0):
    ts1 = time.time()
    (m,d) = X.shape
    D_t = np.ones(m)*1.0/m
    h_list = [] # list of h_i
    a_list = [] # list of corresponding weights a_i
    prevR = np.zeros(m)
    prevA = 0
    prev_r_tilde = 0
    
    for t in range(T):
        h_t = baseLearner(**kwargs)
        h_t.fit(X,y,sample_weight = D_t)
        yhx_t = y * h_t.predict(X)
        res = minimize(G, [0.5], args = (prevR,prevA,yhx_t,nu,prev_r_tilde,True,r_lb),bounds = [(0.01,1.0)], tol=1e-6)
        a_t = res.x[0]
        # R_t, A_t = updateRho(prevR,prevA, a_t, yhx_t)
        R_t = prevR + a_t * yhx_t # un-normalized
        A_t = prevA + a_t
        r_t = R_t/A_t
        r_opt_t = G(a_t,prevR,prevA,yhx_t,nu,prev_r_tilde,False,r_lb)
        xi_t = positivePart(r_opt_t - r_t) 
        if verbose>1: print("sum xi_t = ",np.sum(xi_t), "r_opt_t",r_opt_t,"margin error P_S[yf(x) < r_opt_t] = ", np.sum(R_t/A_t < r_opt_t)*1.0/m)
        prev_r_tilde = r_opt_t - np.sum(xi_t)/(nu*m)
        D_t = D_t * np.exp(-(r_t + xi_t))
        D_t = D_t / np.sum(D_t)

        prevR = np.copy(R_t)
        prevA = A_t
        
        h_list.append(h_t)
        a_list.append(a_t)
        # if verbose>1:
        #     # print ("----------- round:",t,"-----------")
        #     # print("Feature index:",np.argmax(h_t.feature_importances_),"\t a_t:",a_t)
        #     # i_1,i_2 = (0,10)
        #     # # print ("some D_t", D_t[i_1:i_2])
        #     # # print ("some xi_t", xi_t[i_1:i_2])
        #     # # print ("some r_t", r_t[i_1:i_2])
        #     # print ("margin error P_S[y * f(x) < r_opt] = ", np.sum(R_t/A_t < r_opt_t)*1.0/m , "compare to nu",nu, "r_opt:", r_opt_t)
        if verbose>0 and t == T-1: print("final r_opt",r_opt_t)
    
    if verbose>0: print("Time spent in roBoost: {0:.2f} seconds".format(time.time()-ts1))
    return {'h':h_list,'a':a_list}

# sparsification ######################################################################################
def sampleHypotheses(f,k):
    # sample k out of h_i with probability proportional to a_i, with replacement
    # weight of sampled h_j set to 1/k
    T = len(f['a'])
    ind = np.random.choice(np.arange(T),size=k,p=f['a']/np.sum(f['a']))
    h_list = []
    a_list= []
    for i in ind:
        h_list.append(f['h'][i])
        a_list.append(1.0/k)
    return {'h':h_list,'a':a_list}

def repeatedSampleHypotheses(f,k,X,y,B,beta,verbose = 0):
    # repeat the sampling process B times and get the best g on training data
    # B = m * log (1/delta) so that with prob > 1-delta, there exists g st. P_S[yg(x)<beta] <= E_Q[P_S[yg(x)<beta]]
    # beta chosen as some point between cut-off theta and 2 theta, eg, theta, beta=1.5 theta, gamma = rho = 2 theta
    
    best_g = {}
    best_e = 1.0
    h_list = []
    a_list= []
    _sum_e = 0
    ts1 = time.time()
    if verbose>1: print ("full classifier's margin error P_S[ yf(x) < {0:.2f}] = {1:.4f} ".format(beta, marginError(X,y,f,theta = beta) ))
    for t in range(B):
        g = sampleHypotheses(f,k)
        e = marginError(X,y,g,theta = beta)
        _sum_e = _sum_e + e
        if verbose>1: print("sample repetition:",t, "current sampled g has P_S[ yg(x) < {0:.4f}] = {1:.4f} ".format(beta,e), "prevsiou best margin error = {0:.4f}, average so far = {1:.4f}".format(best_e,_sum_e/(t+1)))
        if (e < best_e):
            best_g = {'h':g['h'],'a':g['a']}
            best_e = e
    if verbose>0: print("Time spent in repeatedSampleHypotheses: {0:.2f} seconds.".format(time.time()-ts1))
    return best_g


# deRandomize sampling based on conditional expectation
def binCdf(B, n, p):
    s = 0
    for t in range(1,math.ceil(B)):
        s += math.comb(n,t)*math.pow(p,t)*math.pow(1-p,n-t)
    return s

def deRandomizeSampleHypotheses(f,k,X,y,beta,verbose = 0):
    # T = len(f['h'])
    # m = X.shape[0]
    # aHy = np.zeros((m,T))
    # a_sum = np.sum(f['a'])
    
    # p = np.zeros(m)
    # for t in range(T):
    #     aHy[:,t] = f['a'][t]/a_sum * f['h'][t].predict(X) * y # +/- a's
    # for i in range(m):
    #     for t in range(T):
    #         if (aHy[i][t]>0):
    #             p[i] += aHy[i][t]
    ts1 = time.time()
    y_pred = weightedVote(f,X)
    a_plus = 0.5*(y*y_pred + 1) # for a sampled h, y_i * h(x_i) = 1 with prob a_plus[i]
    # print (a_plus[:30]-p[:30],(y_pred*y)[:30],p[:30])

    T = len(f['h'])
    m = X.shape[0]
    h_list = []
    a_list= []
    prev_yHx = np.zeros(m)
    # repeat k-1 times:
    for t in range(1,k): # t = 1 ... k-1
        # find argmin_h [\sum_{i=1}^{m} P_Q[V_i <= beta_i]] where V_i ~ Bin(k-1,a_plus[i]), beta_i = (k*beta - y_i * h(x_i)+k-1)/2
        best_h_ind = -1
        best_val = 2*m # some large value
        for j in range(T):
            h = f['h'][j]
            yHx = y * h.predict(X) # a vector of the correctness of this h
            # beta_list = np.floor(0.5*(k*beta - yHx - prev_yHx + k-t)).astype(int)
            beta_list = 0.5*(k*beta - yHx - prev_yHx + k-t)
            s = 0
            for i in range(m):
                s += binCdf(beta_list[i],k-t,a_plus[i]) # cdf of P[V_i <= beta_i]
            if (s<best_val): # this current h is better
                best_val = s
                best_h_ind = j
            if (verbose>1) and (j%25==0): print("during round ", t, " checking h_",j, " whose conditional expectation is ", s/m, "best so far:",best_h_ind,best_val/m)

        if verbose>1 : print("round ", t, " selected h_",best_h_ind, " whose conditional expectation is ", best_val/m)
                
        h_list.append(f['h'][best_h_ind])
        a_list.append(1.0/k)
        prev_yHx += y * f['h'][best_h_ind].predict(X)
    # the last one
    best_h_ind = -1
    best_val = 2*m # some large value
    for j in range(T):
        h = f['h'][j]
        yHx = y * h.predict(X) + prev_yHx
        s = 0
        for i in range(m):
            if (yHx[i]< k*beta): s+=1
                
        if (s<best_val): # this current h is better
            best_val = s
            best_h_ind = j
        if (verbose>1)  and (j%25==0): print("during round ", k, " checking h_",j, " whose conditional expectation is ", s/m, "best so far:",best_h_ind,best_val/m)

    if verbose>1 : print("round ", k, " selected h_",best_h_ind, " whose avg Indicator{yg(x)<= k*beta} is ", best_val/m)        
    h_list.append(f['h'][best_h_ind])
    a_list.append(1.0/k)
    if verbose>0: print("Time spent in deRandomizeSampleHypotheses: {0:.2f} seconds".format(time.time()-ts1))
    return {'h':h_list,'a':a_list}

# output ######################################################################################

def plot_margin_distribution(X_train,y_train,f,theta,name='f'):
    # plot the margin distribution
    
    fig, ax = plt.subplots()
    conf_margin_f_train = y_train*weightedVote(f,X_train)
    # conf_margin_f_test = y_test*weightedVote(f,X_test)
    # geo_margin_f_test = np.abs(conf_margin_f_test)
    ax.ecdf(conf_margin_f_train,c = '#1f89dc',label="{} train confidence margin".format(name))
    # ax.ecdf(conf_margin_f_test,c = '#dc721f',label="f test confidence margin")
    # ax.ecdf(geo_margin_f_test,c = '#dc721f',label="f test geometric margin",linestyle="dashed")
    ax.set_title("Margin distribution of "+name, loc = 'left')
    ax.set_xlabel("y * {}(x)".format(name))
    # ax2.set_xlim(xmin=-1,xmax=1)
    # ax2.vlines(theta,0,1, linestyles='dashed',colors=['#1FC11F'], label='theta')
    
    ax.legend()
    ax.set_ylim([0, 1])
    
    ax.vlines(theta,0,1, linestyles='dashed',colors=['#1FC11F'], label='theta')
    
    fig.set_figheight(4)
    fig.set_figwidth(13)
    plt.show()

    
def compute_error_ERROR_cover_margin(X_train,y_train,X_test,y_test,g,k,ref_theta,blackbox):
    margins = []
    covers_train = []
    errors_train = []
    ERRORS_train = []
    covers_test = []
    errors_test = []
    ERRORS_test = []
    ref_used = False
    pred_train = weightedVote(g,X_train)
    pred_test = weightedVote(g,X_test)
    pred_bb_train = blackbox.predict(X_train) 
    pred_bb_test = blackbox.predict(X_test) 
    for i in range(0,k+1,1):
        theta = 1.0*i/k
        if ref_theta<=theta and ref_used == False:
            margins.append(ref_theta)
            ref_used = True
        if(ref_theta!= theta): margins.append(theta)
    
    for theta in margins:
        ind = np.arange( y_train.shape[0])[np.abs(pred_train)>=theta]
        u_ind = np.arange( y_train.shape[0])[np.abs(pred_train)<theta]
        c = ind.shape[0]*1.0/y_train.shape[0]
        covers_train.append(c)
        if c>0:
            error = np.sum(pred_train[ind]*y_train[ind] <= 0)*1.0/ind.shape[0]
        else:
            error = 0
        errors_train.append(error)
        ERROR = (np.sum(pred_train[ind]*y_train[ind] <= 0) + np.sum(pred_bb_train[u_ind]*y_train[u_ind] <= 0))*1.0/y_train.shape[0]
        ERRORS_train.append(ERROR)
        # print ("train ERROR:",ERROR,"k",theta,"error",error)

        ind = np.arange( y_test.shape[0])[np.abs(pred_test)>=theta]
        u_ind = np.arange( y_test.shape[0])[np.abs(pred_test)<theta]
        c = ind.shape[0]*1.0/y_test.shape[0]
        
        covers_test.append(c)
        if c>0:
            error = np.sum(pred_test[ind]*y_test[ind] <= 0)*1.0/ind.shape[0]
        else:
            error = 0
        errors_test.append(error)
        ERROR = (np.sum(pred_test[ind]*y_test[ind] <= 0) + np.sum(pred_bb_test[u_ind]*y_test[u_ind] <= 0))*1.0/y_test.shape[0]
        ERRORS_test.append(ERROR)
    return np.stack([errors_train,ERRORS_train,covers_train,errors_test,ERRORS_test,covers_test,margins], axis=0)
    

def plot_ERROR_cover_by_margin(k_m_ee_cc,ref_theta,ref_error=None,title="Error and coverage VS margin (averaged over n_fold folds)",band=False,file_name=None):
# ['k',  0
# 'margin', 1
# 'train_error_mean', 2
# 'train_error_std', 3
# 'train_ERROR_mean', 4
# 'train_ERROR_std', 5
# 'train_cover_mean', 6
# 'train_cover_std', 7
# 'test_error_mean', 8
# 'test_error_std', 9
# 'test_ERROR_mean', 10
# 'test_ERROR_std', 11
# 'test_cover_mean', 12
# 'test_cover_std'] 13
    colors = ['#1f89dc','#dc721f',"#7d3dad","#1f4e5f","#6b9998","#8cbed6","#000000","#ed872d"]
    margins = k_m_ee_cc[:,1] 
    errors_mean = k_m_ee_cc[:,8] 
    ERRORS_mean = k_m_ee_cc[:,10]
    covers_mean = k_m_ee_cc[:,12] 

    height_width = (4,6)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ls1 = "-"
    ls2 = "--"
    ls3 = ':'
    
    ax1.plot(margins,errors_mean,c = colors[0],label='Interpretable Error',linestyle=ls1)
    ax1.plot(margins,ERRORS_mean,c = colors[2],label='Overall Error',linestyle=ls1)
    ax2.plot(margins,covers_mean,c = colors[1],label='Coverage',linestyle=ls2)
    if ref_error is not None:
        error_bb_mean = np.mean(ref_error,axis=0)
        # error_bb_std = np.std(ref_error,axis=0)
        ax1.hlines(error_bb_mean,margins[0],margins[-1], linestyles=ls3,colors=['#000000'], label='Blackbox error')

    ax2.vlines(ref_theta,0,1, linestyles=ls3,colors=['#1FC11F'], label='Cut-off margin')
    if band: 
        errors_std = k_m_ee_cc[:,3+4]
        covers_std = k_m_ee_cc[:,5+4]
        ax1.fill_between(margins, errors_mean - errors_std, errors_mean + errors_std, color=colors[0], alpha=0.2)   
        ax2.fill_between(margins, covers_mean - covers_std, covers_mean + covers_std, color=colors[1], alpha=0.2)   

    ax1.legend(loc=3)
    ax2.legend(loc=1)
    ax1.set_title(title, loc = 'left')
    ax1.set_xlabel("Cut-off margin")
    ax1.set_ylabel("Error")
    ax2.set_ylabel("Coverage")
    
    # ax1.set_ylim([0,1.2*np.max(errors_mean)])
    # ax2.set_ylim([0,1.0])
    
    fig.set_figheight(height_width[0])
    fig.set_figwidth(height_width[1])
    if file_name!= None: plt.savefig('output/'+file_name+'.png')
    plt.show()
    
def plot_error_by_cover(k_m_ee_cc_list,labels,title="Error VS coverage (averaged over n_fold folds)",band=True,file_name=None,y_lim = False,color_ls=None):
# ['k',  0
# 'margin', 1
# 'train_error_mean', 2
# 'train_error_std', 3
# 'train_ERROR_mean', 4
# 'train_ERROR_std', 5
# 'train_cover_mean', 6
# 'train_cover_std', 7
# 'test_error_mean', 8
# 'test_error_std', 9
# 'test_ERROR_mean', 10
# 'test_ERROR_std', 11
# 'test_cover_mean', 12
# 'test_cover_std'] 13
    if color_ls is None:
        colors = ['#3498db','#1f618d',"#2e4053","#1f4e5f","#079c94","#6b9998","#000000"]
        ls = ['solid','dashed','dotted','dashdot','dashdotted','densely dashdotted']
    else:
        colors = color_ls['c']
        ls = color_ls['l']
    height_width=(4,6)
    fig, ax = plt.subplots()
    y_max = 0
    for i in range(len(k_m_ee_cc_list)):
        k_m_ee_cc = k_m_ee_cc_list[i]
        errors_mean = k_m_ee_cc[:,8] #6
        covers_mean = k_m_ee_cc[:,12] #8
        _y_max = errors_mean[0]
        ax.plot(covers_mean,errors_mean,c = colors[i],label=labels[i],linestyle=ls[i])
        if band: 
            errors_std = k_m_ee_cc[:,11] #7
            _y_max += errors_std[0]
            ax.fill_between(covers_mean, errors_mean - errors_std, errors_mean + errors_std, color=colors[i], alpha=0.2)
        if y_max<_y_max:
            y_max = _y_max
    ax.legend(loc=2)
    ax.set_title(title, loc = 'left')
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Error")

    if y_lim:
        ax.set_ylim([-0.01,y_max*1.05])
    
    fig.set_figheight(height_width[0])
    fig.set_figwidth(height_width[1])
    if file_name!= None: plt.savefig('output/'+file_name+'.png')
    plt.show()




