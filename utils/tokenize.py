#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import transformers
from tokenizers import BertWordPieceTokenizer

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   create a tokenizer to create tokens of the text input.                                      #
#                                                                                               #
#***********************************************************************************************#
def tokenizer(model="BERT-1"):
    # check which model we are using and tokenize text data accordingly
    if model == "BERT-1":
        # First load the real tokenizer
        tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        # Save the loaded tokenizer locally
        tokenizer.save_pretrained('.')
        # Reload it with the huggingface tokenizers library
        fast_tokenizer = BertWordPieceTokenizer("vocab.txt", lowercase=False)
        # return the newly created tokenizer
        return tokenizer, fast_tokenizer