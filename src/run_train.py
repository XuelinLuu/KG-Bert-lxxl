import torch
import os
import transformers
import torch.nn .functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers.file_utils import WEIGHTS_NAME
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import config
from datasets import KGProcessor
from engine import train_fn
from models import BertKG
def run_train():
    data_dir = config.DATA_DIR
    kgp = KGProcessor()
    rela_list = kgp.get_all_relations()
    examples = kgp.get_train_examples(data_dir)
    tokenizer = transformers.BertTokenizer.from_pretrained(config.BERT_TOKENIZER_PATH)
    features = kgp.convert_examples_to_features(examples, config.MAX_SEQ_LEN, tokenizer)

    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    labels = torch.tensor([f["label"] for f in features])
    labels = F.one_hot(labels)
    labels = torch.tensor(labels.numpy(), dtype=float)

    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.TRAIN_BATCH_SIZE
    )

    num_training_steps = len(input_ids) / config.TRAIN_BATCH_SIZE * config.TRAIN_EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertKG(config.BERT_MODEL_PATH, len(rela_list))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(config.TRAIN_EPOCHS):
        print(f"\n---------------------------epoch: {epoch+1}---------------------------")
        train_fn(model, device, data_loader, optimizer, scheduler)

        model_to_save = model.module if hasattr(model, "module") else model
        output_path = os.path.join(f"{config.BERT_OUTPUT_PATH}/{epoch+1}", WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_path)
        tokenizer.save_vocabulary(f"{config.BERT_OUTPUT_PATH}/{epoch+1}/vocab.txt")

    model_to_save = model.module if hasattr(model, "module") else model
    output_path = os.path.join(f"{config.BERT_OUTPUT_PATH}", WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_path)
    tokenizer.save_vocabulary(f"{config.BERT_OUTPUT_PATH}/vocab.txt")

if __name__ == '__main__':
    run_train()