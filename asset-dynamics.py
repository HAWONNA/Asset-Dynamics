#!/usr/bin/env python
# coding: utf-8

# ## Asset Dynamics 
# 
# ![image.png](attachment:47b38635-9a72-49a7-b2ff-8f25eb12a0da.png)
# 
# ![image.png](attachment:6ebe1cf6-acc1-4d5c-b25b-bda1ec162493.png)

# ## GBM
# ![image.png](attachment:90d0a333-c61e-4e7d-8514-dc5522f64011.png)
# 
# 
# ![image.png](attachment:a6d46e11-f00d-401b-9552-1a9df87d0803.png)
# 
# 
# ![image.png](attachment:15ef5e73-b7a6-452d-8d05-ca5de77db4c0.png)
# 
# 
# ![image.png](attachment:f9ec2d4c-c878-4ec3-b93f-190df35fee69.png)
# 
# 
# ![image.png](attachment:ecfcb727-3df9-468f-b9e4-17999ae64bec.png)
# 

# In[1]:


get_ipython().system('pip install pandas-datareader')


# In[2]:


import pandas as pd 
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Adj Price
df = web.get_data_yahoo('META', start = '2020-01-01', end='2020-12-31')['Adj Close']
df


# In[4]:


#Adj return 
ret = df.pct_change()
ret = ret.dropna()
ret


# In[5]:


# log price 

log_price = np.log(df)
log_price


# In[6]:


# log return 

log_return = log_price.shift(-1) - log_price
log_return = log_return.dropna()
log_return


# In[7]:


# V = E[log return] = E[lnSt-lnSt-1]

V = log_return.mean() # t변화량 = 하루 

# std = std[log return]

std = np.sqrt(log_return.var()) # t변화량 = 하루 

print(V)
print(std)


# In[8]:


# t변화량 - 하루 

t_change = 1 

# 100일 예측 

t_value = np.arange(1, 101, 1) #start, stop, step 
t_value


# In[9]:


# 0~1 값부여 -> 정규분포 cdf의 inverse 값들로부여 

import scipy as sp
import scipy.stats

rv = sp.stats.norm()

xx = np.linspace(-8, 8, 100)
cdf = rv.cdf(xx)
plt.plot(xx, cdf)


# In[10]:


Epsilon = np.random.choice(cdf, size=t_value.shape)


# In[11]:


# Epsilon 여러개 생성 

Epsilon_list = [] 

for i in range(10000): 
    Eps = np.random.standard_normal(t_value.shape[0])
    Epsilon_list.append(Eps)

Epsilon_list[0].shape


# ![image.png](attachment:d9895fee-71a0-4e8e-a0c3-1476df9d95be.png)

# In[12]:


# 난수값 부여 log_return 

log_return_senario_list = [] 
price_return_senario_list = []

for Eps in Epsilon_list:
    log_return_senario = V*t_change + std*Eps*np.sqrt(t_change)
    log_return_senario = pd.Series(log_return_senario, index= t_value)
    log_return_senario_list.append(log_return_senario)
    
    price_return_senario = np.exp(log_return_senario) - 1 
    price_return_senario_list.append(price_return_senario)
    
    


# In[13]:


log_return_senario_list[0]  # v+s*Error


# In[14]:


price_return_senario_list[0] # e^(v+s*Error)


# In[15]:


# 10000개 시나리오 return들의 expected return, standarad deviation of return 구하기 

price_expected_return_list= []
price_risk_list = []
log_expected_return_list =[]
log_risk_list =[]
for price_return, log_return in zip(price_return_senario_list, log_return_senario_list):
    
    ann_price_return = price_return.mean() * 252
    ann_price_risk = np.sqrt(price_return.var() * 252)
    ann_log_return = log_return.mean() * 252
    ann_log_risk = np.sqrt(log_return.var() * 252)
    
    price_expected_return_list.append(ann_price_return)
    price_risk_list.append(ann_price_risk)
    log_expected_return_list.append(ann_log_return)
    log_risk_list.append(ann_log_risk)
    
    


# In[16]:


# 구한 10000개 시나리오 return들의 expected return, standarad deviation of return 들의 percentile값들 구하기 


# price return, risk 오름차순 정렬 

price_expected_return_list.sort()
price_risk_list.sort()
log_expected_return_list.sort()
log_risk_list.sort()

# 하위 10% return,risk

lower_10_price_return = np.percentile(price_expected_return_list, 0.1)
lower_10_price_risk = np.percentile(price_risk_list, 0.1)
lower_10_log_return = np.percentile(log_expected_return_list, 0.1)
lower_10_log_risk = np.percentile(log_risk_list, 0.1)



# 중위 50% return,risk
middle_price_return = np.percentile(price_expected_return_list, 0.5)
middle_price_risk = np.percentile(price_risk_list, 0.5)
middle_log_return = np.percentile(log_expected_return_list, 0.5)
middle_log_risk = np.percentile(log_risk_list, 0.5)

# 상위 10% return,risk

upper_10_price_return = np.percentile(price_expected_return_list, 0.9)
upper_10_price_risk = np.percentile(price_risk_list, 0.9)
upper_10_log_return = np.percentile(log_expected_return_list, 0.9)
upper_10_log_risk = np.percentile(log_risk_list, 0.9)






print('price')
print('<return>')
print(lower_10_price_return, middle_price_return, upper_10_price_return)
print('<risk>')
print(lower_10_price_risk, middle_price_risk, upper_10_price_risk)
print('\n')

print("all senarios' price returns mean", np.mean(price_expected_return_list)) #이미 annaulized 되어있는 값들 -> 다시해줄필요없음 
print("all senarios' price returns std", np.sqrt(np.var(price_expected_return_list)))

print('\n')


print('log_price')
print('<return>')
print(lower_10_log_return, middle_log_return, upper_10_log_return)
print('<risk>')
print(lower_10_log_risk, middle_log_risk, upper_10_log_risk)

print('\n')


print("all senarios' log_price returns mean", np.mean(log_expected_return_list)) #이미 annaulized 되어있는 값들 -> 다시해줄필요없음 
print("all senarios' log_price returns std", np.sqrt(np.var(log_expected_return_list)))




# ## Return,risk Distribution 파악
# 
# ##### 모든 return senario들의 expected return, std of return 들의 분포를 파악 

# ### Return 

# In[17]:


import seaborn as sns
import scipy as sp 

sns.distplot(price_expected_return_list, kde=True, rug= True, fit=sp.stats.norm)
sns.distplot(log_expected_return_list, kde=True, rug= True, fit=sp.stats.norm)

sns.set(rc={'figure.figsize':(20,20)})





# In[18]:


# skewness -price_return_senarios'

sp.stats.skew(price_expected_return_list), sp.stats.kurtosis(price_expected_return_list)


# In[19]:


# skewness -log_return_senarios'

sp.stats.skew(log_expected_return_list), sp.stats.kurtosis(log_expected_return_list)


# ### Risk  

# In[20]:


import seaborn as sns
import scipy as sp 

sns.distplot(price_risk_list, kde=True, rug= True, fit=sp.stats.norm)
sns.distplot(log_risk_list, kde=True, rug= True, fit=sp.stats.norm)

sns.set(rc={'figure.figsize':(20,20)})





# In[21]:


# skewness - 각 price_return senario들의 std of return 들 

sp.stats.skew(price_risk_list), sp.stats.kurtosis(price_risk_list)


# In[22]:


# skewness - 각 log_return senario들의 std of return 들 

sp.stats.skew(log_risk_list), sp.stats.kurtosis(log_risk_list)


# # Price 예측 

# ### 1개 Senario

# In[23]:


# v+s*Error (dt=1) -> expected S1 = S0 * e^(v+s*Error)
# expected S2 = expected S2 + e^(v+s*Error)

# 보통 마지막 price를 S0으로 놓고 진행!!!! 

exp_return = np.exp(log_return_senario_list[0]) # e^(v+s*Error)
exp_return


# In[24]:


# p_senario = df * np.exp(log_return_senario_list[0]) 잘못된 예시!! 

price_expected = []
price_expected.append(df[-1])

for i in range(len(exp_return)):
    price_expect = price_expected[i] * exp_return[i+1]
    price_expected.append(price_expect)


# In[25]:


plt.plot(t_value, price_expected[:-1])


# ## 10000개 Senarios

# In[26]:


exp_return_list = []

for log_return_senario in log_return_senario_list:

    exp_return = np.exp(log_return_senario) # e^(v+s*Error)
    exp_return_list.append(exp_return)


# In[27]:


price_expected_list = []

for exp_return in exp_return_list:
    price_expected = []
    price_expected.append(df[-1])

    for i in range(len(exp_return)):

        price_expect = price_expected[i] * exp_return[i+1]
        price_expected.append(price_expect)
    price_expected_list.append(price_expected)


# In[28]:


for i in range(len(price_expected_list)):
    plt.plot(t_value, price_expected_list[i][:-1])


# ## Applying on Option market

# # Call option, long 가정 

# In[29]:


# T=100 시점이 option 만기일 
# T=100 시점 S(T)와 F(T)비교 

# S(0) * 1.5 = F(T)

Ft = df[-1] * 1.2 # S(0) = historical price 마지막
Opt_price = df[-1] * 0.1 #옵션사는비용 - 이번엔 고려 x  

option_return = []

for price_expect in price_expected_list:
    if price_expect[-1] > Ft : # S(T) > Ft 이면 optin 행사 
        opt_return = (price_expect[-1] - Ft)/ Ft 
    else:  # S(T) <= Ft 이면 optin 행사 x 
        opt_return = 0
    
    option_return.append(opt_return)


# ## Option 행사한 return들의 distribution 

# In[30]:


(np.array(option_return)==0).sum()


# In[31]:


#Option 행사안한 비율 

print('option 행사한 개수: ', (np.array(option_return)==0).sum())
print('option 행사하지 않은 개수 : ', len(option_return)- (np.array(option_return)==0).sum())
print('option 행사비율: ', (np.array(option_return)==0).sum()/ len(option_return)) 


# In[35]:


option_exe_return = np.array(option_return)[np.array(option_return)!=0]


# In[36]:


import seaborn as sns
import scipy as sp 

sns.distplot(option_exe_return, kde=True, rug= True, fit=sp.stats.norm)
sns.set(rc={'figure.figsize':(20,20)})


# In[39]:


print(np.mean(option_exe_return), np.sqrt(np.var(option_exe_return)))


# In[38]:


# 향후 performance 진행할 수 있음 (MDD, CVaR, 다양한 Ratio 등)

