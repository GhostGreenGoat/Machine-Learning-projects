#%%
import numpy as np
import matplotlib.pyplot as plt

def load_mean(file):
    with open(file,'r') as f:
        meanVecotr = []
        for line in f:
            line=line.strip('\n')
            line=line.split()
            meanVecotr.append(line)
    return meanVecotr

def load_hip(file):
    with open(file,'r') as f:
        hip = []
        for line in f:
            line=line.strip('\n')
            line=line.split(',')
            hip.append(line)
    return hip      
   
def distance(x,y):
    return np.linalg.norm(x-y)

def assigneClassMember(meanVecotr,hip):
    classMember=[]
    for i in range(len(hip[0])):
        hipsample=hip[:,i]
        minDistance=distance(meanVecotr[:,0],hipsample)
        Category=0
        for j in range(len(meanVecotr[0])):
            classMean=meanVecotr[:,j]
            dist=distance(classMean,hipsample)
            if dist<minDistance:
                minDistance=dist
                Category=j
        classMember.append((i,Category))
    return classMember
            
def reAssumeMean(hip,classMember,meanVecotr):
    #calculate the mean of each class
    newMean=np.zeros((len(meanVecotr),len(meanVecotr[0])))
    for k in range(len(meanVecotr[0])):
        classNumber=0
        for i in range(len(classMember)):
            if classMember[i][1]==k:
                newMean[:,k]=newMean[:,k]+hip[:,classMember[i][0]]
                classNumber=classNumber+1
        if classNumber!=0:
            newMean[:,k]=newMean[:,k]/classNumber
    return newMean

def isKmeansStop(meanVecotr,newMean):
    #K-means stop when the mean vector does not change
    if np.array_equal(meanVecotr,newMean):
        return True
    else:
        return False

def kmeans(hip,meanVecotr):
    iteration=0
    objectiveValue=[]
    while True:
        #K-means stop when the mean vector does not change
        classMember=assigneClassMember(meanVecotr,hip)
        newMean=reAssumeMean(hip,classMember,meanVecotr)
        obj=objectiveFunction(hip,classMember,meanVecotr)
        objectiveValue.append(obj)
        if isKmeansStop(meanVecotr,newMean):
            break
        meanVecotr=newMean
        
        
        iteration=iteration+1
    print('K-means stop after '+str(iteration)+' iterations.')
    #print('The final mean vector is:',newMean)
    #print('old mean vector is:',meanVecotr)
    return classMember,objectiveValue
        

def objectiveFunction(hip,classMember,meanVector):
    #calculate the objective function for k-means
    obj=0
    for i in range(len(classMember)):
        obj=obj+(distance(hip[:,classMember[i][0]],meanVector[:,classMember[i][1]]))**2
    return obj


    
#%%
#load the data
mean=load_mean('test_mean.txt')
mean=np.array(mean)
mean=np.array(mean,dtype=float)
hip=load_hip('hip1000.txt')
hip=np.array(hip)
hip=np.array(hip,dtype=float)
hipName=load_hip('hip1000names.txt')

# %%
#problem 4a
finalClassMember,objectiveValue=kmeans(hip,mean)
classResult=[0,0,0]
for i in range(len(finalClassMember)):
    if finalClassMember[i][1]==0:
        classResult[0]=classResult[0]+1
    elif finalClassMember[i][1]==1:
        classResult[1]=classResult[1]+1
    else:
        classResult[2]=classResult[2]+1
print('The number of samples in each class is:',classResult)

# %%
#plot the objective over iterations
iteration=[1,2,3,4]
plt.plot(iteration,objectiveValue,marker='o')
plt.xticks(iteration)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.title('Objective Value over Iterations')
plt.show()
# %%
#problem 4b
#Run K-means algorithm with K = 3 with your own random initialization
mean=np.random.rand(208,3)
finalClassMember,objectiveValue=kmeans(hip,mean)
# %%
#Plot the correlation coefficient matrix of the raw data

def getCorrelationMatrix(hip):
    #calculate the correlation coefficient matrix
    matrix=np.zeros((len(hip[0]),len(hip[0])))
    for i in range(len(hip[0])):
        for j in range(len(hip[0])):
            matrix[i,j]=np.corrcoef(hip[:,i],hip[:,j])[0,1]
    return matrix

#%%

TestMatrix=getCorrelationMatrix(hip)
# %%
#plot the correlation coefficient matrix
plt.imshow(TestMatrix)
plt.colorbar()
plt.title('Correlation Coefficient Matrix for the Raw Data')
plt.show()
# %%
#group the columns according to the clusters found by K-means
def getGroupedHip(hip,finalClassMember):
    #group the columns according to the clusters found by K-means
    groupHip=np.zeros((len(hip),len(finalClassMember)))
    sortedClassMember=sorted(finalClassMember,key=lambda x:x[1])
    for i in range(len(sortedClassMember)):
        groupHip[:,i]=hip[:,sortedClassMember[i][0]]
    return groupHip

# %%
groupedHip=getGroupedHip(hip,finalClassMember)
groupedMatrix=getCorrelationMatrix(groupedHip)
# %%
#plot the correlation coefficient matrix
plt.imshow(groupedMatrix)
plt.colorbar()
plt.title('Correlation Coefficient Matrix for the Grouped Data')
plt.show()
# %%
#plot the correlation coefficient matrix int the same plot
fig,ax=plt.subplots(1,2)
ax[0].imshow(TestMatrix)
ax[0].set_title('Raw Data')
ax[1].imshow(groupedMatrix)
ax[1].set_title('Grouped Data')
plt.savefig('4b'+'.pdf',format='pdf')
plt.show()
# %%
#problem 4c
time=0
groupedHips=[]
obValues=[]
while time<10:
    mean=np.random.rand(208,3)
    finalClassMember,objectiveValue=kmeans(hip,mean)
    groupedHip=getGroupedHip(hip,finalClassMember)
    groupedHips.append(groupedHip)
    obValues.append(objectiveValue[-1])
    time=time+1
#%%
groupedMatices=[]
for i in range(len(groupedHips)):
    groupedMatrix=getCorrelationMatrix(groupedHips[i])
    groupedMatices.append(groupedMatrix)

# %%
#plot 10 correlation coefficient matrices
fig,ax=plt.subplots(2,5,figsize=(20,10),sharex=True,sharey=True)
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(groupedMatices[i*5+j])
        ax[i,j].set_title('Objective Value:'+str(format(obValues[i*5+j],'.2f')))
plt.savefig('4c'+'.pdf',format='pdf')
plt.show()
# %%
#problem 4d
#Run K-means algorithm for k=3 to k=12 with 10 random initializations for mean vectors
#and plot the objective function over iterations for each k

def run10kmeans(k,iteration):
    time=0
    groupedHips=[]
    obValues=[]
    while time<iteration:
        mean=np.random.rand(208,k)
        finalClassMember,objectiveValue=kmeans(hip,mean)
        groupedHip=getGroupedHip(hip,finalClassMember)
        groupedHips.append(groupedHip)
        obValues.append(objectiveValue[-1])
        time=time+1
    return obValues,groupedHips


# %%
obValues,groupedHips=run10kmeans(3,10)
# %%
obvalues4,groupedHips4=run10kmeans(4,10)
# %%
obvalues5,groupedHips5=run10kmeans(5,10)
# %%
obvalues6,groupedHips6=run10kmeans(6,10)
# %%
obvalues7,groupedHips7=run10kmeans(7,10)
# %%
obvalues8,groupedHips8=run10kmeans(8,10)
# %%
obvalues9,groupedHips9=run10kmeans(9,10)
obvalues10,groupedHips10=run10kmeans(10,10)
obvalues11,groupedHips11=run10kmeans(11,10)
obvalues12,groupedHips12=run10kmeans(12,10)
# %%
#plot the best objective value for each k
bestObValues=[min(obValues),min(obvalues4),min(obvalues5),min(obvalues6),min(obvalues7),min(obvalues8),min(obvalues9),min(obvalues10),min(obvalues11),min(obvalues12)]
bestObIndex=[obValues.index(min(obValues)),obvalues4.index(min(obvalues4)),obvalues5.index(min(obvalues5)),obvalues6.index(min(obvalues6)),obvalues7.index(min(obvalues7)),obvalues8.index(min(obvalues8)),obvalues9.index(min(obvalues9)),obvalues10.index(min(obvalues10)),obvalues11.index(min(obvalues11)),obvalues12.index(min(obvalues12))]
k=[3,4,5,6,7,8,9,10,11,12]
#%%
#plot 12 correlation coefficient matrices
groupedMatrices3=getCorrelationMatrix(groupedHips[bestObIndex[0]])
groupedMatrices4=getCorrelationMatrix(groupedHips4[bestObIndex[1]])
groupedMatrices5=getCorrelationMatrix(groupedHips5[bestObIndex[2]])
groupedMatrices6=getCorrelationMatrix(groupedHips6[bestObIndex[3]])
groupedMatrices7=getCorrelationMatrix(groupedHips7[bestObIndex[4]])
groupedMatrices8=getCorrelationMatrix(groupedHips8[bestObIndex[5]])
groupedMatrices9=getCorrelationMatrix(groupedHips9[bestObIndex[6]])
groupedMatrices10=getCorrelationMatrix(groupedHips10[bestObIndex[7]])
groupedMatrices11=getCorrelationMatrix(groupedHips11[bestObIndex[8]])
groupedMatrices12=getCorrelationMatrix(groupedHips12[bestObIndex[9]])
# %%
#plot k=3 to k=12 correlation coefficient matrices
fig,ax=plt.subplots(2,5,figsize=(20,10),sharex=True,sharey=True)
for i in range(2):
    for j in range(5):
        if i*5+j==0:
            ax[i,j].imshow(groupedMatrices3)
            ax[i,j].set_title('k=3'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))
        elif i*5+j==1:
            ax[i,j].imshow(groupedMatrices4)
            ax[i,j].set_title('k=4'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))
        elif i*5+j==2:
            ax[i,j].imshow(groupedMatrices5)
            ax[i,j].set_title('k=5'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))
        elif i*5+j==3:
            ax[i,j].imshow(groupedMatrices6)
            ax[i,j].set_title('k=6'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))
        elif i*5+j==4:
            ax[i,j].imshow(groupedMatrices7)
            ax[i,j].set_title('k=7'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))
        elif i*5+j==5:
            ax[i,j].imshow(groupedMatrices8)
            ax[i,j].set_title('k=8'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))
        elif i*5+j==6:
            ax[i,j].imshow(groupedMatrices9)
            ax[i,j].set_title('k=9'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))
        elif i*5+j==7:
            ax[i,j].imshow(groupedMatrices10)
            ax[i,j].set_title('k=10'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))
        elif i*5+j==8:
            ax[i,j].imshow(groupedMatrices11)
            ax[i,j].set_title('k=11'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))
        elif i*5+j==9:
            ax[i,j].imshow(groupedMatrices12)
            ax[i,j].set_title('k=12'+' Objective Value:'+str(format(bestObValues[i*5+j],'.2f')))

plt.savefig('4d'+'.pdf',format='pdf')
plt.show()
# %%
