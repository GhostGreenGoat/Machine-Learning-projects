#%%
import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()
    """
    train_input="./en_data/train.txt"
    index_to_word="./en_data/index_to_word.txt"
    index_to_tag="./en_data/index_to_tag.txt"

    train_data = list()
    with open(train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices

#%%
if __name__ == "__main__":
    # Collect the input data

    # Initialize the initial, emission, and transition matrices

    # Increment the matrices

    # Add a pseudocount

    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter=" " for the matrices)
    
    #initialization
    train_data, words_to_indices, tags_to_indices = get_inputs()
    initial=np.zeros((len(tags_to_indices),1))
    emission=np.zeros((len(tags_to_indices),len(words_to_indices)))
    transition=np.zeros((len(tags_to_indices),len(tags_to_indices)))

    #Increment the matrices based on counts and add pseudocount
    for sentence in train_data:
        for i in range(len(sentence)):
            if i==0:
                initial[tags_to_indices[sentence[i][1]]]+=1
            else:
                transition[tags_to_indices[sentence[i-1][1]],tags_to_indices[sentence[i][1]]]+=1
            emission[tags_to_indices[sentence[i][1]],words_to_indices[sentence[i][0]]]+=1
    #add pseudocount
    initial+=1
    emission+=1
    transition+=1

    #calculate probabilities
    initial=initial/np.sum(initial)
    emission=emission/np.sum(emission,axis=1).reshape(-1,1)
    transition=transition/np.sum(transition,axis=1).reshape(-1,1)

    #save to files
    np.savetxt("./en_data/hmmprior.txt",initial,delimiter=" ")
    np.savetxt("./en_data/hmmemit.txt",emission,delimiter=" ")
    np.savetxt("./en_data/hmmtrans.txt",transition,delimiter=" ")

            
            


# %%
