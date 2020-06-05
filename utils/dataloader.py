#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import tensorflow as tf
import transformers
from tokenizers import BertWordPieceTokenizer

import importlib
encoders = importlib.import_module("encoders")
tokenize = importlib.import_module("tokenize")

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define Global Configuration Variables                                                       #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
AUTO = tf.data.experimental.AUTOTUNE

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   create a dataloader to load the toxic comment classification data.                          #
#                                                                                               #
#***********************************************************************************************#
def dataLoad(path="../../data/", model="BERT-1", BATCH_SIZE=32, MAX_LEN=192):
    # check which model we are using and load data accordingly
    if model == "BERT-1":
        # LOADING THE DATA
        train1 = pd.read_csv(path+"jigsaw-toxic-comment-train.csv")
        valid = pd.read_csv(path+"validation.csv")
        test = pd.read_csv(path+"test.csv")
        sub = pd.read_csv(path+"sample_submission.csv")
    
    # apply the pre-processing step ont he loaded dataset
    x_train, x_valid, x_test, y_train, y_valid = preprocess(model=model, train1, valid, test, maxlen=MAX_LEN)
    
    """
    Finally create datasets from each of the obtained chunks    
    """
    
    # Training dataset
    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    # Validation dataset
    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )

    # Testing dataset
    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(BATCH_SIZE)
    )
    
    # Return the newly created datsets
    return train_dataset, valid_dataset, test_dataset


#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   create a module for any sort of data pre-processing required.                               #
#                                                                                               #
#***********************************************************************************************#
def preprocess(model="BERT-1", train1, valid, test, maxlen=192)
    # check which model we are using and pre-process data accordingly
    if model == "BERT-1":    
        # create an instant of the tokenizer
        _ , fast_tokenizer = tokenize.tokenizer(model="BERT-1")
        # pre-process the data by tokenizing and then encoding it
        x_train = encoders.data_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=maxlen)
        x_valid = encoders.data_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=maxlen)
        x_test = encoders.data_encode(test.content.astype(str), fast_tokenizer, maxlen=maxlen)

        y_train = train1.toxic.values
        y_valid = valid.toxic.values
        
    return x_train, x_valid, x_test, y_train, y_valid