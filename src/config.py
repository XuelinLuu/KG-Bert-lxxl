DATA_DIR = "../datasets"

TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 5e-5
MAX_SEQ_LEN = 128


BERT_MODEL_PATH = "../bert-base-tiny-uncased"
BERT_TOKENIZER_PATH = "../bert-base-tiny-uncased/vocab.txt"
BERT_OUTPUT_PATH = "../models"
PREDICTION_OUTPUT_PATH = "../predictions/kg.tsv"
THRESHOLD = 0.7