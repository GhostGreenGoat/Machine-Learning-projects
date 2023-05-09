#%%
import sys
import numpy as np
import math
import collections

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self,colname):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.type= None
        self.level=0
        self.colname= colname
        self.label=None

def predict(node,example,colname):
    if node.type=='leaf':
        return node.vote
    else:
        #print("node.attr",node.attr)
        #print("node.colname.index(node.attr)",node.colname.index(node.attr))
        if example[colname.index(node.attr)]==1:
            return predict(node.left,example,colname)
        if example[colname.index(node.attr)]==0:
            return predict(node.right,example,colname)


def tree(df,colname,maxdepth):
    root=trainDecisionTree(df,colname=colname,level=0,maxdepth=maxdepth)
    return root


def trainDecisionTree(df,colname,level,maxdepth):
    '''
    This function will train a decision tree on the given data frame (nparray without colnames) and
    return the root node of the tree.
    colname is a list of the column names of the data frame.
    '''
    root=Node(colname)
    root.level=level
    label=df[-1]
    root.label=label

    if len(df)==0:
        root.vote=None
        root.type='leaf'
        print("empty data")
        return root

    elif len(set(label))==1:
        root.vote=label[0]
        root.type='leaf'
        print("all same label")
        return root
    
    elif len(df)>0:
        attributes_number=len(df)-1
        #print("attributes_number",attributes_number)
        for i in range(len(df)-2):
            #print("i",i)
            if (df[i]==df[i+1]).all():
                attributes_number-=1
            #print("attributes_number",attributes_number)
        if attributes_number==0:
            root.vote=collections.Counter(label).most_common(1)[0][0]
            root.type='leaf'
            #print("all same attributes")
            return root

    if root.level<maxdepth:
        root.attr=split(df,root.colname)
        if root.attr!=None:
            root.type='internal'
            left_df,right_df,new_colname=getNewdf(df,root.colname.index(root.attr),root.colname)
            #root.colname=new_colname
            root.level+=1
            root.left=trainDecisionTree(left_df,new_colname,root.level,maxdepth)
            root.right=trainDecisionTree(right_df,new_colname,root.level,maxdepth)
        else:
            #print("df",df)
            #print("colname",colname)
            #print("label",label)
            root.vote=collections.Counter(label).most_common(1)[0][0]
            root.type='leaf'
            #print("remaining attributes' entropy less than 0!")
            return root
    else:
        root.vote=collections.Counter(label).most_common(1)[0][0]
        root.type='leaf'
        #print("max depth reached!")
        return root
    return root


#==============================================================================    
#calulate entropy
#input: Y is a list of labels
#output: entropy of Y, a float. Probability of each label is calculated by its frequency
def entropy(Y):
  df=collections.Counter(Y)
  entropy=0.0
  for key in df.keys():
    df[key]=df[key]/len(Y)
    entropy+=df[key]*math.log(df[key],2)
  return -entropy

#get the labels conditions on attribute X
#input: df is the data frame, X is the index of corresponding attribute in df, Y is the index of label
#output: a dictionary, whose key is the value of attribute X, the value is the probability of each x value.
# a dictionary, whose key is the value of attribute X, and value is the list of labels corresponding to the key .
def conditionalLabel(df,X,Y):
  p=collections.Counter(df[X])
  conditional={}
  for key in p.keys():
    p[key]=p[key]/len(df[X])
    condlist=[df[X]==key]
    choicelist=[df[Y]]
    list=np.select(condlist, choicelist, -1)
    conditional[key] = np.delete(list, np.where(list == -1))
  return p,conditional

#calculate conditional entropy
#input: df is the data frame, X is the index of corresponding attribute in df, Y is the index of label
#output: conditional entropy of Y given X, a float
def conditionalEntropy(df,X,Y):
  p,conditional=conditionalLabel(df,X,Y)
  conditionalEntropy=0.0
  for key in p.keys():
    conditionalEntropy+=p[key]*entropy(conditional[key])
  return conditionalEntropy

#calulate the mutual information of Y given X
def mutualInformation(df,X,Y):
  return entropy(df[Y])-conditionalEntropy(df,X,Y) 

#get the new data frames after removing the attribute X
#input: df is the data frame, X is the index of corresponding attribute in df, colnames is the list of column names
#output: new_df_left is the new data condition on X=1, new_df_right is the new data condition on X=0, colnames is the list of column names
def getNewdf(df,X,colnames):
  new_df_left=[]
  new_df_right=[]
  colnames=np.array(colnames)
  element=set(df[X])
  element=list(element)
  condlist_left=[df[X]==element[1]]
  condlist_right=[df[X]==element[0]]
  colnames=np.delete(colnames,X)
  for i in range(len(df)):
    if i!=X:
      choicelist=[df[i]]
      l=np.select(condlist_left, choicelist, -1)
      l=np.delete(l, np.where(l==-1))
      new_df_left.append(l)
      l=np.select(condlist_right, choicelist, -1)
      l=np.delete(l, np.where(l==-1))
      new_df_right.append(l)
  new_df_left=np.array(new_df_left)
  new_df_right=np.array(new_df_right)
  return new_df_left,new_df_right,list(colnames)

def split(df,colname):
    '''
    This function will return the attribute that will be used to split the data
    frame on the next iteration of the decision tree algorithm.
    '''
    label=len(colname)-1
    maxGain=0
    maxCol=0
    for col in range(len(colname)-1):
        tmpGain=mutualInformation(df,col,label)
        if tmpGain>maxGain:
            maxGain=tmpGain
            maxCol=col
    if maxGain>0:
        return colname[maxCol]
    else:
        return None
    

#%%

def read_tsv(path):
    with open(path) as f:
        lines = f.readlines()
        for line in range(len(lines)):
            if line == 0:
                columns = lines[line].strip().split('\t')
                df=[[] for c in range(len(columns))]
                continue
            else:
                lines[line] = lines[line].strip().split('\t')
                for column in range(len(lines[line])):
                    df[column].append(int(lines[line][column]))
        df=np.array(df)
    return columns, df


# %%
#==============================================================================
def predict_all(df,node,colname):
    predict_result=[]
    for i in range(len(df[0])):
        example=[]
        for j in range(len(df)-1):
            example.append(df[j][i])
        #print(f"example {i} is {example}")
        predict_result.append(predict(node,example,colname))
    return predict_result

def evaluate(df,predict):
    error=0
    label=df[-1]
    for i in range(len(label)):
        if predict[i]!=label[i]:
            error+=1
    return error/len(label)

def write_tsv(train_path,test_path,metics_path,train_predict,test_predict,train_error,test_error):
    with open(train_path,'w') as f:
        for i in range(len(train_predict)):
            f.write(str(train_predict[i])+'\n')
    with open(test_path,'w') as f:
        for i in range(len(test_predict)):
            f.write(str(test_predict[i])+'\n')
    with open(metics_path,'w') as f:
        f.write("error(train): "+str(train_error)+'\n')
        f.write("error(test): "+str(test_error)+'\n')

#%%
# =============================================================================
def printTree(node):
    if node.type=='leaf':
        return
    else:
        #count=collections.Counter(node.label)
        #print(f"[{count[0]} 0/{count[1]} 1]")
        if node.right!=None:
            print("| "*node.level + node.attr + " = 0: ",end='')
            rightcount=collections.Counter(node.right.label)
            print(f"[{rightcount[0]} 0/{rightcount[1]} 1]")
            printTree(node.right)
        if node.left!=None:
            print("| "*node.level + node.attr + " = 1: ",end='')
            leftcount=collections.Counter(node.left.label)
            print(f"[{leftcount[0]} 0/{leftcount[1]} 1]")
            printTree(node.left)

    return

#%%
zero=collections.Counter(train_df[-1])
print(f"[{zero[0]} 0/{zero[1]} 1]")
printTree(Tree)


#%%

train_input="./handout/heart_train.tsv"
test_input="./handout/heart_test.tsv"
maxdepth=3
output_train="education_train.labels.txt"
output_test="education_test.labels.txt"
output_metrics="education_metrics.txt"

train_colname,train_df = read_tsv(train_input)
test_colname,test_df = read_tsv(test_input)


Tree=tree(train_df,train_colname,maxdepth)


train_predict=predict_all(train_df,Tree,train_colname)
train_error=evaluate(train_df,train_predict)

test_predict=predict_all(test_df,Tree,test_colname)
test_error=evaluate(test_df,test_predict)
print("train_error: ",round(train_error,4))
print("test_error: ",round(test_error,4))

#write_tsv(output_train,output_test,output_metrics,train_predict,test_predict,train_error,test_error)

#%%
#test functions 

print(test_predict)
#%%
#plot
import matplotlib.pyplot as plt
trainError=[]
testError=[]
maxDepth=[]
for m in range(len(train_colname)-1):
    Tree=tree(train_df,train_colname,m)
    train_predict=predict_all(train_df,Tree,train_colname)
    train_error=evaluate(train_df,train_predict)
    trainError.append(train_error)
    test_predict=predict_all(test_df,Tree,test_colname)
    test_error=evaluate(test_df,test_predict)
    testError.append(test_error)
    maxDepth.append(m)

plt.plot(maxDepth,trainError,label='train')
plt.plot(maxDepth,testError,label='test')
plt.xlabel('maxDepth')
plt.ylabel('error rate')
plt.legend()
#%%
if __name__ == '__main__':
    train_input=sys.argv[1]
    test_input=sys.argv[2]
    maxdepth=int(sys.argv[3])
    output_train=sys.argv[4]
    output_test=sys.argv[5]
    output_metrics=sys.argv[6]

    train_colname,train_df = read_tsv(train_input)
    test_colname,test_df = read_tsv(test_input)


    Tree=tree(train_df,train_colname,maxdepth)

    train_example=train_df[:-1]
    trian_predict=predict_all(train_example,Tree)
    train_error=evaluate(train_df,trian_predict)
    
    test_example=test_df[:-1]
    test_predict=predict_all(test_example,Tree)
    test_error=evaluate(test_df,test_predict)
    write_tsv(output_train,output_test,output_metrics,trian_predict,test_predict,train_error,test_error)

#%%


