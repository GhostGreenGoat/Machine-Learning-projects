#%%
import argparse
import numpy as np

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)
    
    args = parser.parse_args()
    """
    hmminit="./en_data/hmmprior.txt"
    hmmemit="./en_data/hmmemit.txt"
    hmmtrans="./en_data/hmmtrans.txt"
    index_to_word="./en_data/index_to_word.txt"
    index_to_tag="./en_data/index_to_tag.txt"
    validation_input="./en_data/validation.txt"
    predicted_file="./en_data/predicted.txt"
    metric_file="./en_data/metrics.txt"

    validation_data = list()
    with open(validation_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans,predicted_file, metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix

def forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix

        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)

    # Initialize log_alpha and fill it in - feel free to use words_to_indices to index the specific word
    log_alpha = np.zeros((L, M))
    for t in range(0, L):
        for j in range(M):
            if t == 0:
                log_alpha[t,j] = loginit[j] + logemit[j][words_to_indices[seq[t]]]
                #print("log_alpha,t=0:",log_alpha[t,j])

            else:
                log_alpha[t,j] = logsumexp(log_alpha[t-1,:] + logtrans[:,j]) + logemit[j,words_to_indices[seq[t]]]
    #print(log_alpha)
    # Initialize log_beta and fill it in - feel free to use words_to_indices to index the specific word
    log_beta = np.zeros((L, M))
    for t in range(L-1, -1, -1):
        for j in range(M):
            if t==L-1:
                log_beta[t][j] = 0
            else:
                log_beta[t][j] = logsumexp(log_beta[t+1] + logtrans[j,:] + logemit[:,words_to_indices[seq[t+1]]])

    # Compute the predicted tags for the sequence - tags_to_indices can be used to index to the rwquired tag
    #tags_to_indices is dictionary
    predicted_tags = []
    path=list()
    for t in range(L):
        path.append(np.argmax(log_alpha[t]+log_beta[t]))
        for key,value in tags_to_indices.items():
            if value==path[t]:
                predicted_tags.append(key)



    # Compute the stable log-probability of the sequence
    log_prob = logsumexp(log_alpha[-1])

    # Return the predicted tags and the log-probability
    return predicted_tags, log_prob, log_alpha, log_beta

    
def logsumexp(x):
    """
    Computes the log-sum-exp trick on a vector or matrix. If x is a vector, it computes the log-sum-exp
    trick on the vector. If x is a matrix, it computes the log-sum-exp trick on the rows of the matrix.
    """
    if len(x.shape) == 1:
        # Compute the log-sum-exp trick on a vector
        m=np.max(x)
        new_x = m + np.log(np.sum(np.exp(x-m)))
        return new_x
    else:
        # Compute the log-sum-exp trick on the rows of a matrix
        m=np.max(x, axis=1)
        new_x = m + np.log(np.sum(np.exp(x-m[:, None]), axis=1))
        return new_x
 #%%   
    
if __name__ == "__main__":
    # Get the input data
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans,predicted_file,metric_file = get_inputs()
    hmminit=np.log(hmminit)
    hmmemit=np.log(hmmemit)
    hmmtrans=np.log(hmmtrans)

    
    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.
    log_prob_list = list()
    predicted_tags_list = list()
    for sequence in validation_data:
        #extract the words from the sequence
        words = [pair[0] for pair in sequence]
        #print(words)
        predicted_tags, log_prob,log_alpha,log_beta = forwardbackward(words, hmminit, hmmtrans, hmmemit, words_to_indices, tags_to_indices)
        predicted_tags_list.append(predicted_tags)
        log_prob_list.append(log_prob)

    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    average_log_likelihood = np.mean(log_prob_list)
    total_accuracy = 0
    word_counts = 0
    for seq in range(len(validation_data)):
        for word in range(len(validation_data[seq])):
            word_counts += 1
            if validation_data[seq][word][1] == predicted_tags_list[seq][word]:
                total_accuracy += 1

    total_accuracy=total_accuracy/word_counts

    print("Average Log-Likelihood: " + str(average_log_likelihood))
    print("Accuracy: " + str(total_accuracy))
    
    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood: " + str(average_log_likelihood) + "\n")
        f.write("Accuracy: " + str(total_accuracy) + "\n")
    """

    with open(predicted_file, "w") as g:
        for seq in range(len(validation_data)):
            for word in range(len(validation_data[seq])):
                g.write(validation_data[seq][word][0] + "\t" + validation_data[seq][word][1] + "\t" + predicted_tags_list[seq][word] + "\n")
            g.write("\n")
    """

# %%
