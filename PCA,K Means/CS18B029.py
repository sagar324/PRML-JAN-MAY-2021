#!/usr/bin/env python
# coding: utf-8

# # General Instructions to students:
# 
# 1. There are 5 types of cells in this notebook. The cell type will be indicated within the cell.
#     1. Markdown cells with problem written in it. (DO NOT TOUCH THESE CELLS) (**Cell type: TextRead**)
#     2. Python cells with setup code for further evaluations. (DO NOT TOUCH THESE CELLS) (**Cell type: CodeRead**)
#     3. Python code cells with some template code or empty cell. (FILL CODE IN THESE CELLS BASED ON INSTRUCTIONS IN CURRENT AND PREVIOUS CELLS) (**Cell type: CodeWrite**)
#     4. Markdown cells where a written reasoning or conclusion is expected. (WRITE SENTENCES IN THESE CELLS) (**Cell type: TextWrite**)
#     5. Temporary code cells for convenience and TAs. (YOU MAY DO WHAT YOU WILL WITH THESE CELLS, TAs WILL REPLACE WHATEVER YOU WRITE HERE WITH OFFICIAL EVALUATION CODE) (**Cell type: Convenience**)
#     
# 2. You are not allowed to insert new cells in the submitted notebook.
# 
# 3. You are not allowed to **import** any extra packages.
# 
# 4. The code is to be written in Python 3.6 syntax. Latest versions of other packages maybe assumed.
# 
# 5. In CodeWrite Cells, the only outputs to be given are plots asked in the question. Nothing else to be output/print. 
# 
# 6. If TextWrite cells ask you to give accuracy/error/other numbers you can print them on the code cells, but remove the print statements before submitting.
# 
# 7. Any runtime failures on the submitted notebook as it is will get zero marks.
# 
# 8. All code must be written by yourself. Copying from other students/material on the web is strictly prohibited. Any violations will result in zero marks.
# 
# 9. The dataset is given as .npz file, and will contain data in  numpy array. 
# 
# 10. All plots must be labelled properly, all tables must have rows and columns named properly.
# 
# 11. You are allowed to use the numpy library to calculate eigen values. All other functions for reconstruction, clustering, etc., should be written from scratch.
# 
# 12. Change the name of the file with your roll no.
# 
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
data = np.load('Data.npz')
data2 =data['arr_0']
#data2.shape[0]


# **Cell type : TextRead**
# 
# # Problem 5: Life in Lower Dimensions
# 
# You are provided with a dataset of 1797 images - each image is 8x8 pixels and provided as a feature vector of length 64. You will try your hands at transforming this dataset to a lower-dimensional space using PCA, and perform K-means clustering the images in this reduced space.
#  
# 
# 
# 

# In[ ]:


# Cell type : CodeWrite
# write the function for PCA and K-means clustering here. 
def PCA(M, D, X):
    X_bar = np.mean(X,axis = 0)[None, :]
    S = np.dot(X.T-X_bar.T, X-X_bar)/len(X)
    eigenvals, eigenvecs = np.linalg.eig(S)
    #plt.plot(eigenvals)
    N = len(X)
    U = np.array(eigenvecs)
    Z = np.zeros((N,M))
    b = np.zeros(D)
    for n in range(N):
        for i in range(M):
            Z[n,i] = np.dot(X[n], U[: , i])
    for i in range(M,D):
        b[i] = np.dot(X_bar, U[: , i])
    
    X_tilde = np.zeros(X.shape)
    
    for n in range(N):
        for i in range(M):
            X_tilde[n, : ] += Z[n,i]*U[: , i].T
        for i in range(M,D):
            X_tilde[n, : ] += b[i]*U[: , i].T
    
    return [X_tilde,Z]

def dist(x,y):
    l = len(x)
    result = 0
    for i in range(l):
        result += (x[i]-y[i])*(x[i]-y[i])
    return result

def K_means(X,k):
    x = X.tolist()
    N = len(x)
    cols = len(x[0])
    Means=[]
    clusters=[]
    for i in range(k):
        Means.append(x[i])
    i = 0
    while 1:
        i = i+1
        lst = []
        for j in range(k):
            lst.append([])
        for j in range(N):
            dst = []
            for itr in range(k):
                dst.append(dist(x[j],Means[itr]))
            itr=0
            mini = dst[0]
            minindex=0
            while itr<k:
                if dst[itr]<mini:
                    mini = dst[itr]
                    minindex = itr
                itr = itr + 1
            lst[minindex].append(x[j])
        clusters = lst
        newMeans=[]
        for j in range(k):
            result = []
            for itr in range(cols):
                result.append(0)
            for itr in range(len(lst[j])):
                for itr2 in range(cols):
                    result[itr2] += lst[j][itr][itr2]
            for itr in range(cols):
                result[itr] = result[itr]/len(lst[j])
            newMeans.append(result)
        if newMeans == Means:
            break
        Means = newMeans
    error = 0
    for i in range(k):
        for j in range(len(clusters[i])):
            error += dist(Means[i],clusters[i][j])
    return [Means,clusters,error]


# 

# **Cell type : TextRead**
# 
# # Problem 5
# 
# #### 5a) Run PCA algorithm on the given data-set. Plot the cumulative percentage variance explained by the principal components. Report the number of principal components that contribute to 90% of the variance in the dataset.
# 
# 
# 

# In[ ]:


# Cell type : CodeWrite
# write the code for loading the data, running the PCA algorithm, and plotting. 
# (Use the functions written previously.)
X = data2
X_bar = np.mean(X,axis = 0)[None, :]
S = np.dot(X.T-X_bar.T, X-X_bar)/len(X)
eigenvals, eigenvecs = np.linalg.eig(S)

yaxis = []
xaxis = []
for i in range(1,65):
    xaxis.append(i)
dp = []
for i in range(1,65):
    dp.append(0)
dp[0] = eigenvals[0]
for i in range(1,64):
    dp[i] = dp[i-1]+eigenvals[i]
sumeigen = np.sum(eigenvals)
dp = dp/sumeigen
dp = dp*100
yaxis = dp
index = 0
for i in range(64):
    if yaxis[i]>=90:
        index = i
        break

plt.xlabel("Number of principal components")
plt.ylabel("% Variance in the dataset")
plt.scatter(xaxis,yaxis)
plt.show()


# ####5b)  Perform reconstruction of data using the dimensionality-reduced data considering the number of dimensions [2,4,8,16]. Report the Mean Square Error (MSE) between the original data and reconstructed data, and interpret the optimal dimensions $\hat{d}$ based on the MSE values.
# 
# 

# In[ ]:


# Cell type : CodeWrite
# Write the code for data reconstruction, run the algorithm for dimensions.
xaxis=[2,4,8,16]
yaxis=[]
for itr in range(4):
    M = xaxis[itr]
    output = PCA(M,64,X)
    X2 = output[0]
    Xlst = X.tolist()
    X2lst = X2.tolist()
    error=0
    for i in range(1797):
        error += dist(Xlst[i],X2lst[i])
        pass
    error = error/1797
    yaxis.append(error)


plt.xlabel("Number of reduced Dimensions")
plt.ylabel("MSE")
plt.scatter(xaxis,yaxis)
plt.show()



# ####5c) Apply K-means clustering on the reduced dataset from last subpart (b) (i.e., the $R^{64}$ to $R^\hat{d}$ reduced dataset; pick the initial k points as cluster centers during initialization). Report the optimal choice of K you have made from the set [1...15]. Which method did you choose to find the optimum number of clusters? And explain briefy why you chose that method. Also, show the 2D scatter plot (consider only the first two dimensions of optimal $\hat{d}$) of the datapoints based on the cluster predicted by K-means (use different color for each cluster).
# 

# In[ ]:


# Cell type : CodeWrite
# Write the code for dimensionality reduction, run k-means algorithm on the reduced data-set and do plotting.
output = PCA(16,64,X)
Z = output[1]
xaxis = []
yaxis = []
for itr in range(1,16):
    xaxis.append(itr)
    output2 = K_means(Z,itr)
    error = output2[2]
    yaxis.append(error/1797)
plt.subplot(1,2,1)
plt.plot(xaxis,yaxis,'bx-')
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of squared errors")
#optimal k is 9
output2 = K_means(Z,9)
clusters = output2[1]
Means = output2[0]
plt.subplot(1,2,2)
plt.xlabel("Dimension -1")
plt.ylabel("Dimension -2")

colors2 = ["lightsalmon","grey","red","yellow","deeppink","purple","darkkhaki","royalblue","lime"]


for i1 in range(9):
    for j1 in range(len(clusters[i1])):
        x = []
        y = []
        x.append(clusters[i1][j1][0])
        y.append(clusters[i1][j1][1])
        #plt.scatter(x, y,s = 10,color = colors[2*i1+2])
        plt.scatter(x, y,s=10,color = colors2[i1] )
x = []
y = []
for i1 in range(9):
    x.append(Means[i1][0])
    y.append(Means[i1][1])
    plt.scatter(x,y,s=30,color = "black",marker = 'X')

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=2.0, 
                    top=0.9,
                    wspace=0.4, 
                    hspace=0.4)
plt.show()


# ####5d) In the next TextWrite cell, Summarise and explain your observations from the above experiments. Is the PCA+K-means clustering consistent with how your brain would cluster the images?
# 
# 

# **Cell type : TextWrite**
# 
# Report your observations, MSE values and the method used to choose the optimal number of clusters.
# 
# 5a)In 5a,If the number of principal components is which contribute to 90%  in the dataset  is 21, which we showed using a plot in 5a).
# 
# 5b)By considering the number of dimensions as [2,4,8,16],the MSE between the original data and the reconstructed data were found to be [858.9447808487345, 616.1911300562695, 391.7947361149766, 180.93970325737882]
# Therefore as we can see by taking 16 dimensions ,we get the minimum Mean Square error,so reducing dimensions to 16 is more optimal in [2,4,8,16].
# 
# 5c)We plot the Sum of squared errors(divided by 1797) as we modify the number of clusters,to analyse the optimum number of clusters.
# If we look at the first graph we can see at the point 9(number of clusters =9),the decrease in error is high and thereafter the SSE decreases little.(i.e the elbow is at 9 clusters)
# Hence ,9 clusters are the optimum number of clusters.
# 
# 5d)Yes the PCA+K-means clustering is consistent with what my brain would cluster the images , as we can see the we figured out the optimum number of clusters were 9/10 which are the blurred digits images present in the database.
# We can also see that 9 clusters were formed properly with some overlap ,but the plots were of only 2 dimensions,so the overlapped parts are actually clustered well in other dimensions as well.
