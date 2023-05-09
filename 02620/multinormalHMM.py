#%%
import numpy as np
from matplotlib import pyplot as plt
import time
import random

def MultiVarNormal(x,mean,cov):
    """
    MultiVarNormal implements the PDF for a mulitvariate gaussian distribution
    (You can do one sample at a time of all at once)
    Input:
        x - An (d) numpy array
            - Alternatively (n,d)
        mean - An (d,) numpy array; the mean vector
        cov - a (d,d) numpy arry; the covariance matrix
    Output:
        prob - a scaler
            - Alternatively (n,)

    Hints:
        - Use np.linalg.pinv to invert a matrix
        - if you have a (1,1) you can extrect the scalar with .item(0) on the array
            - this will likely only apply if you compute for one example at a time
    """
    det = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = X - mean
    exponent = -0.5 * np.einsum('ij,ij->i', diff @ inv_cov, diff)
    return (1.0 / np.sqrt((2 * np.pi) ** X.shape[1] * det)) * np.exp(exponent)

def UpdateMixProps(hidden_matrix):
    """
    Returns the new mixing proportions given a hidden matrix
    Input:
        hidden_matrix - A (n, k) numpy array
    Output:
        mix_props - A (k,) numpy array
    Hint:
        - See equation in Lecture 10 pg 42

    """
    n, k = hidden_matrix.shape
    mix_props = np.sum(hidden_matrix, axis=0) / n
    return mix_props

def UpdateMeans(X, hidden_matrix):
    """
    Returns the new means for the gaussian distributions given the data and the hidden matrix
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix - A (n, k) numpy array
    Output:
        new_means - A (k,d) numpy array
    Hint:
        - See equation in Lecture 10 pg 43
    """
    n, d = X.shape
    k = hidden_matrix.shape[1]
    new_means = np.zeros((k, d))
    for i in range(k):
        # Compute the weighted sum of X for the i-th Gaussian
        weighted_X_sum = np.sum(X * hidden_matrix[:, i].reshape(n, 1), axis=0)
        # Compute the sum of the weights for the i-th Gaussian
        weight_sum = np.sum(hidden_matrix[:, i])
        # Compute the new mean for the i-th Gaussian
        new_means[i, :] = weighted_X_sum / weight_sum
    return new_means

def UpdateCovar(X, hidden_matrix_col, mean):
    """
    Returns new covariance for a single gaussian distribution given the data, hidden matrix, and distribution mean
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix - A (n, k) numpy array
        mean - A (d,) numpy array; the mean for this distribution
    Output:
        new_cov - A (d,d) numpy array
    Hint:
        - See equation in Lecture 10 pg 43
    """
    n, d = X.shape
    diff = X - mean
    new_cov = np.zeros((d, d))
    for i in range(n):
        new_cov += hidden_matrix_col[i] * np.outer(diff[i, :], diff[i, :])
    new_cov /= np.sum(hidden_matrix_col)
    return new_cov

def UpdateCovars(X, hidden_matrix, means):
    """
    Returns a new covariance matrix for all distributions using the function UpdateCovar()
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix - A (n, k) numpy array
        u - A (k,d) numpy array; All means for the distributions
    Output:
        new_covs - A (k,d,d) numpy array
    Hint:
        - Use UpdateCovar() function
    """
    k, d = means.shape
    new_covs = np.zeros((k, d, d))
    for i in range(k):
        new_covs[i, :, :] = UpdateCovar(X, hidden_matrix[:, i], means[i, :])
    return new_covs


def HiddenMatrix(X, means, covs, mix_props):
    """
    Computes the hidden matrix for the data. This function should also compute the log likelihood
    Input:
        X - An (n,d) numpy array
        means - An (k,d) numpy array; the mean vector
        covs - a (k,d,d) numpy arry; the covariance matrix
        mix_props - a (k,) array; the mixing proportions
    Output:
        hidden_matrix - a (n,k) numpy array
        ll - a scalar; the log likelihood
    Hints:
        - Construct an intermediate matrix of size (n,k). This matrix can be used to calculate the loglikelihood and the hidden matrix
            - Element t_{i,j}, where i in {1,...,n} and j in {1,k}, should equal
            P(X_i | c = j)P(c = j)
        - Each rows of the hidden matrix should sum to 1
            - Element h_{i,j}, where i in {1,...,n} and j in {1,k}, should equal
                P(X_i | c = j)P(c = j) / (Sum_{l=1}^{k}(P(X_i | c = l)P(c = l)))
    """
    n, d = X.shape
    k = means.shape[0]
    intermediate_matrix = np.zeros((n, k))
    for i in range(k):
        intermediate_matrix[:, i] = mix_props[i] * MultiVarNormal(X, means[i, :], covs[i, :, :])
    hidden_matrix = intermediate_matrix / np.sum(intermediate_matrix, axis=1).reshape(n, 1)
    ll = np.sum(np.log(np.sum(intermediate_matrix, axis=1)))
    return hidden_matrix, ll



def GMM(X, init_means, init_covs, init_mix_props, thres=0.001):
    """
    Runs the GMM algorithm
    Input:
        X - An (n,d) numpy array
        init_means - a (k,d) numpy array; the initial means
        init_covs - a (k,d,d) numpy arry; the initial covariance matrices
        init_mix_props - a (k,) array; the initial mixing proportions
    Output:
        - clusters: and (n,) numpy array; the cluster assignment for each sample
        - ll: th elog likelihood at the stopping condition
    Hints:
        - Use all the above functions
        - Stoping condition should be when the difference between your ll from 
            the current iteration and the last iteration is below your threshold
    """
    #k = init_means.shape[0]
    prev_ll = -np.inf
    lls=[]
    while True:
        # E-step
        hidden_matrix, ll = HiddenMatrix(X, init_means, init_covs, init_mix_props)
        if np.abs(ll - prev_ll) < thres:
            break
        prev_ll = ll
        lls.append(ll)
        # M-step
        cluster_probs = np.sum(hidden_matrix, axis=0)
        init_mix_props = cluster_probs / X.shape[0]
        init_means = UpdateMeans(X, hidden_matrix)
        init_covs = UpdateCovars(X, hidden_matrix, init_means)
    clusters = np.argmax(hidden_matrix, axis=1)
    return clusters, lls,hidden_matrix

def RandCenter(data,k):
    n = data.shape[0]
    # random k size [] from random centroids
    centroids = [[] for _ in range(k)]
    randpairs = random.sample(range(n),k)
    # order pairs
    OP =  enumerate(randpairs)
    for count, item in OP:
       centroids[count] = data[item]
    return np.array(centroids)
#%%
if __name__ == "__main__":
    #load data
    data = np.loadtxt("./data/mouse-data/hip1000.txt", dtype=np.float32,delimiter=',')
    test_means = np.loadtxt("./data/test_mean.txt").T
    genename=np.loadtxt("./data/mouse-data/hip1000names.txt",dtype=str,delimiter=',')
    print('Data shape:',data.shape)
    print('test_means shape: ',test_means.shape)
    print('genename shape: ',genename.shape)
    #get first 10 mouses data
    X = data.T[:,:10]
    test=test_means[:,:10]
    print('X shape: ',X.shape)
    print('test shape: ',test.shape)

    #initialize parameters
    k = 3
    p_init = [0.3,0.3,0.4]
    mu_init = [test[i] for i in range(3)]
    mu_init=np.array(mu_init)
    covs_init = np.zeros((k,X.shape[1],X.shape[1]))
    for i in range(k):
        covs_init[i] = np.eye(X.shape[1])


    #run GMM
    multivar=MultiVarNormal(X,mu_init[0],covs_init[0])
    #print("multivar: ",multivar.shape)


    clusters,ll,hidden_matrix = GMM(X,mu_init,covs_init,p_init)
    #print('clusters: ',clusters)
    #print('ll: ',ll)

    #Plot the log-likelihood of data log p(Data) over iterations
    plt.plot(ll)
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.show()

    #compute the probability of the first gene to belong to each of the three clusters
    print('Probability of the first gene to belong to each of the three clusters: ',hidden_matrix[0,:])

    #Run the EM algorithm, assuming K = 3, . . . , 10 clusters. 
    # Plot the log-likelihood of the data across different values for K. 
    # What is the optimal value for K?
    k_list=[3,4,5,6,7,8,9,10]
    ll_list=[]
    X = data.T[:,:10]

    
    for k in k_list:
        # Create GMM instance
        covs_init = np.zeros((k,X.shape[1],X.shape[1]))
        for i in range(k):
            covs_init[i] = np.eye(X.shape[1])
        #print("covs_init: ",covs_init[0].shape, len(covs_init))
        pi_init = np.full(k, 1.0 / k)
        
        #print("pi_init: ",pi_init)
        for i in range(k):
    # Initialize random centers
            centers = RandCenter(X, k)
        mu_init =  centers
        mu_init=np.array(mu_init)
        #print("mu_init: ",mu_init)
        print("k",k)
        #print("covs_init",covs_init[0, :, :])
        #p=MultiVarNormal(X, mu_init[0, :], covs_init[0, :, :])
        #print("p",p)
        clusters,ll,hidden_matrix = GMM(X,mu_init,covs_init,pi_init)
        ll_list.append(ll[-1])

    #Plot the log-likelihood of data log p(Data) over iterations
    plt.plot(k_list,ll_list)
    plt.xlabel('K')
    plt.ylabel('Log-likelihood')
    #highlight points
    plt.scatter(k_list,ll_list)
    plt.show()





# %%
#Plot the log-likelihood of data log p(Data) over iterations
plt.plot(k_list,ll_list)
plt.xlabel('K')
plt.ylabel('Log-likelihood')
#highlight points
plt.scatter(k_list,ll_list)
plt.show()
# %%
for l in ll_list:
    print(f'k={k_list[ll_list.index(l)]}, log-likelihood={l}')
# %%
