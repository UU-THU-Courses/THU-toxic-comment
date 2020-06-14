import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

import tensorflow as tf
from sklearn import metrics
from keras.preprocessing import sequence, text
from tensorflow.keras.optimizers import Adam

import transformers
from tokenizers import BertWordPieceTokenizer

# # Configuring TPU's
# 
# For this version of Notebook we will be using TPU's as we have to built a BERT Model

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
train2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

train.drop(train.columns.difference(['id','comment_text','toxic']),axis=1,inplace=True)
train2.drop(train2.columns.difference(['id','comment_text','toxic']),axis=1,inplace=True)
print(train.shape)
print(train2.shape)

print(set(list(train.toxic.values)))
train2['toxic'] = train2['toxic'].apply(lambda x: 1 if x>0.5 else 0)
print(set(list(train2.toxic.values)))
print(train.shape)
#train = train.append(train2, ignore_index=True)
print(train.shape)

#del train2

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

#IMP DATA FOR CONFIG

AUTO = tf.data.experimental.AUTOTUNE


# Configuration
EPOCHS = 5
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 250

# ## Tokenization
# 
# For understanding please refer to hugging face documentation again

# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer

del tokenizer

print("Start encoding the data")
x_train = fast_encode(train.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(validation.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)
print("Finished encoding the data")

y_train = train.toxic.values
y_valid = validation.toxic.values

#del fast_tokenizer
del train
del validation
del test
import gc
gc.collect()

print("Prepaaring Training Dataset")
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

print("Prepaaring Validation Dataset")
valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

print("Prepaaring Testing Dataset")
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(256, activation='relu')(cls_token)
    out = Dropout(0.2) (out)
    out = Dense(1, activation='sigmoid')(out)
    
    model = Model(input_word_ids, out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    
    return model

# ## Starting Training
# 
# If you want to use any another model just replace the model name in transformers._____ and use accordingly

with strategy.scope():
    transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_auc', mode='max', verbose=2, patience=1)


print("Start training 1")
n_steps = x_train.shape[0] // BATCH_SIZE # 4000/strategy.num_replicas_in_sync #x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[es]
)

del x_train
del y_train
del train_dataset
gc.collect()

print("Encode training 2")
x_train = fast_encode(train2.iloc[0:900000,:].comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
y_train = train2.iloc[0:900000,:].toxic.values
print("Preparing Training2 Dataset")
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

print("Start training 2")
n_steps = x_train.shape[0] // BATCH_SIZE #4000/strategy.num_replicas_in_sync #x_train.shape[0] // BATCH_SIZE
train_history2 = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=4,
    callbacks=[es]
)

del x_train
del y_train
del train_dataset
gc.collect()


print("Start training on validation data")
es2 = EarlyStopping(monitor='auc', mode='max', verbose=1, patience=1)
n_steps =  n_steps = x_valid.shape[0] // BATCH_SIZE #4000/strategy.num_replicas_in_sync #x_valid.shape[0] // BATCH_SIZE
train_history_4 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=3,
    callbacks=[es2]
)

sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)