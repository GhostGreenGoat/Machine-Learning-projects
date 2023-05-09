# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
#read in the data
expression=pd.read_csv('expression.txt',sep=' ',header=None)
gene_names=pd.read_csv('gene_list.txt',sep=' ',header=None)
strain_list=pd.read_csv('strain_list.txt',sep=' ',header=None)


# %%
#preprocess the data
expression.index=strain_list[0]
expression=expression[expression.columns[:-1]]
expression.columns=gene_names[0]
snp=pd.read_csv('SNPs.txt',sep='  ',header=None, engine='python')
snp.index=strain_list[0]

y=expression['YCL018W']
X=snp
#mean center the data
y_mean=y-y.mean()
X_mean=X-X.mean()


# %%
#perform MLE regression
class univarateRegression:
    def __init__(self):
        self.w=None

    def fit(self,X,y):
        self.w=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

    def predict(self,X):
        return np.dot(X,self.w)


# %%
def draw_scatter(x,y,xlabel,ylabel,title):
    fig,ax=plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#8da0cb')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#8da0cb')
    plt.scatter(x,y,s=5,c='#e78ac3')
    plt.tick_params(axis='x',colors='#8da0cb')
    plt.tick_params(axis='y',colors='#8da0cb')
    plt.xlabel(xlabel,color='#8da0cb')
    plt.ylabel(ylabel,color='#8da0cb')
    plt.title(title,color='black')
    plt.savefig(title+'.pdf',format='pdf')
    plt.show()

#%%
univarate=univarateRegression()
weight=[]
for snp in X_mean.columns:
    univarate.fit(X_mean[snp].values.reshape(-1,1),y_mean)
    weight.append(univarate.w)
#draw the scatter plot(save as pdf file)
draw_scatter(range(len(weight)),weight,'SNP','Weight','Univariate regression')
#%%




# %%
#Perform ridge regression
class ridgeRegression:
    def __init__(self,sigma,sigma0):
        self.w=None
        self.sigma=sigma
        self.sigma0=sigma0

    def fit(self,X,y):
        alpha=self.sigma/self.sigma0
        self.w=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+alpha*np.identity(X.shape[1])),X.T),y)

    def predict(self,X):
        return np.dot(X,self.w)

# %%
ridge5=ridgeRegression(1,5.0)
ridge5.fit(X_mean,y_mean)
w5=ridge5.w
ridge05=ridgeRegression(1,0.005)
ridge05.fit(X_mean,y_mean)
w05=ridge05.w

# %%
draw_scatter(range(len(w5)),w5,'SNP','Weight for sigma0=5.0','Ridge regression with prior N(0,5.0)')
draw_scatter(range(len(w05)),w05,'SNP','Weight for sigma0=0.5','Ridge regression with prior N(0,0.005)')

# %%
#find the most predictive SNP
def find_max_weight(w):
    max_weight=0
    max_index=0
    for i in range(len(w)):
        if abs(w[i])>max_weight:
            max_weight=abs(w[i])
            max_index=i
    return max_index

max_index5=find_max_weight(w5)
max_index05=find_max_weight(w05)

