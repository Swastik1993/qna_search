# This file is just used for downloading the different models and saving them directly inside the docker image. 
# This is done so that the code does not download the models every time once the images are re-started.

# import the specific libraries
import torch
import tensorflow as tf
import tensorflow_hub as hub
import torch
import transformers
from transformers import *
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

torch_device = 'cpu' # since this is the no_gpu branch CUDA property cannot be set for GPU accereation

# download the bert-large-uncased-whole-word-masking-finetuned-squad model and verify it
print("Downloading bert-large-uncased-whole-word-masking-finetuned-squad pre-trained model")
QA_MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
QA_TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
QA_MODEL.to(torch_device)
QA_MODEL.eval()

# download the facebook/bart-large-cnn model and verify it
print("Downloading facebook bart-large-cnn model")
SUMMARY_TOKENIZER = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
SUMMARY_MODEL = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
SUMMARY_MODEL.to(torch_device)
SUMMARY_MODEL.eval()

# download the monologg/biobert_v1.1_pubmed model and verify it [medical domain specific can also be modified for any other tokenizer model]
print("Downloading biobert_v1.1_pubmed pre-trained model")
para_model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')
para_tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)

print("Done. All pre-trained models loaded into docker image")
