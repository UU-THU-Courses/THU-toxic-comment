import torch
import pandas as pd
import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import transformers as ppb
from transformers import  BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset,DataLoader, RandomSampler, SequentialSampler
from keras.utils import to_categorical

import time
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import seaborn as sns


def format_time(elapsed):
  elapsed_rounded = int(round((elapsed)))
  return str(datetime.timedelta(seconds=elapsed_rounded))


def get_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def tokenize_data(tokenizer, sentences, max_len):
  # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []
  attention_masks = []

# For every sentence...
  for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

  # Convert the lists into tensors.
  return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)





if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



# data
train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub =pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

# unccoment train2.drop
train.drop(train.columns.difference(['id','comment_text','toxic']),axis=1,inplace=True)
#train2.drop(train2.columns.difference(['id','comment_text','toxic']),axis=1,inplace=True)

# uncomment train2['toxic'] =
# Uncomment  train = train.append

print(set(list(train.toxic.values)))
#train2['toxic'] = train2['toxic'].apply(lambda x: 1 if x>0.5 else 0)
#print(set(list(train2.toxic.values)))
print(train.shape)
#train = train.append(train2, ignore_index=True)
print(train.shape)

train_labels=train.toxic.values
dev_labels= validation.toxic.values
#test_labels = test.toxic.values

# For DistilBERT:
tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizer, 'distilbert-base-multilingual-cased')


# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#model = model_class.from_pretrained(pretrained_weights)

max_len = 256
#for sent in train.comment_text:
 #   max_len = max(max_len, len(tokenizer.encode(sent, add_special_tokens=True)))
print('Max sentence length: ', max_len)

# Prepare data
print("Preparing training and development datasets")
batch_size = 32
train_input_ids, train_attention_masks = tokenize_data(tokenizer, train.comment_text, max_len)
train_labels = torch.tensor(train_labels)
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

dev_input_ids, dev_attention_masks = tokenize_data(tokenizer, validation.comment_text, max_len)
dev_labels = torch.tensor(dev_labels)
dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)
dev_dataloader = DataLoader(
            dev_dataset, # The validation samples.
            sampler = SequentialSampler(dev_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

#model
print("Load Pretrained model")
model = BertForSequenceClassification.from_pretrained(
    pretrained_weights, # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
      # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
model.to(device)

optimizer = AdamW(model.parameters(),lr = 2e-5, eps = 1e-8)

epochs = 5
total_steps = len(train_dataloader) * epochs
# learning rate scheduler
scheduler = ppb.get_linear_schedule_with_warmup(optimizer,
         num_warmup_steps = 0,num_training_steps = total_steps)

training_stats = []
total_t0 = time.time()
min_val_loss = 100000  # just a very big value 
print("Start training")
for epoch_i in range(0, epochs):
  print("")
  print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
  print('Training...')

  t0 = time.time()
  total_train_loss = 0
  model.train()

  for step, batch in enumerate(train_dataloader):
    if step % 40 == 0 and not step == 0:
      elapsed = format_time(time.time() - t0)
      print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    model.zero_grad()

    loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, 
                             labels=b_labels)
    total_train_loss += loss.item()

    loss.backward()

    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

  avg_train_loss = total_train_loss / len(train_dataloader)
  training_time = format_time(time.time() - t0)

  print("")
  print("  Average training loss: {0:.2f}".format(avg_train_loss))
  print("  Training epcoh took: {:}".format(training_time))

  print("")
  print("Running Validation...")
  t0 = time.time()

  model.eval()

  total_eval_accuracy = 0
  total_eval_loss = 0
  nb_eval_steps = 0

  for batch in dev_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
      (loss, logits) = model(b_input_ids, token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
    total_eval_loss += loss.item()

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    total_eval_accuracy +=get_accuracy(logits, label_ids)

  avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
  print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

  avg_val_loss = total_eval_loss / len(dev_dataloader)
  validation_time = format_time(time.time() - t0)

  if avg_val_loss < min_val_loss:
    print("Saving the model, validatin loss imporoved from", min_val_loss, "to", avg_val_loss)
    min_val_loss = avg_val_loss
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(".")
    tokenizer.save_pretrained(".")

  print("  Validation Loss: {0:.2f}".format(avg_val_loss))
  print("  Validation took: {:}".format(validation_time))

  training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

pd.set_option('precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
print(df_stats)


# Plot figures
# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.figure(1)
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([x for x in range(1, epochs+1)])
plt.savefig("1.bert_loss.png")


# Plot the learning curve.
plt.figure(2)
#plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Accur.'], 'g-o', label="Validation")

# Label the plot.
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.xticks([x for x in range(1, epochs+1)])
plt.savefig("2.bert_acc.png")


#Test
print("Evaluating the best model")
model = BertForSequenceClassification.from_pretrained(".")
model.to(device)
tokenizer = tokenizer_class.from_pretrained(".")

test_input_ids, test_attention_masks = tokenize_data(tokenizer, test.content, max_len)

#test_labels = torch.tensor(test_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks)
test_dataloader = DataLoader(
            test_dataset, 
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))
model.eval()

predictions , true_labels = [], []

print("len(test_dataloader)", len(test_dataloader))

for batch in test_dataloader:

  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask= batch
  with torch.no_grad():
    outputs =  model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

  logits = outputs[0]
  logits = logits.detach().cpu().numpy()
  #label_ids = b_labels.to('cpu').numpy()

  predictions += logits.tolist()
  #true_labels += label_ids.tolist()

sub['toxic'] = predictions
sub.to_csv('submission.csv', index=False)
