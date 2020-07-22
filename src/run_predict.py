import os
import torch
import transformers
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import config as config
from datasets import KGPredicProcessor
from engine import prdict_fn
from models import BertKG


def run_predict():
    data_dir = config.DATA_DIR
    kgpp = KGPredicProcessor()
    rela_list = kgpp.get_all_relations()
    examples = kgpp.get_examples(data_dir)
    tokenizer = transformers.BertTokenizer.from_pretrained(f"{config.BERT_OUTPUT_PATH}/vocab.txt")
    features = kgpp.convert_examples_to_features(examples, config.MAX_SEQ_LEN, tokenizer)

    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    head_tails = [f["head_tail"] for f in features]

    dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
    data_loader = DataLoader(
        dataset
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertKG(config.BERT_OUTPUT_PATH, len(rela_list))
    model.to(device)

    with open(os.path.join(data_dir, "train.tsv"), "r", encoding="utf-8") as fro, open(config.PREDICTION_OUTPUT_PATH,
                                                                                       "a", encoding="utf-8") as too:
        for line in fro.readlines():
            line = line.rstrip("\n").split("\t")
            too.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")

    prdict_fn(model, device, data_loader, rela_list, head_tails, config.PREDICTION_OUTPUT_PATH, config.THRESHOLD)


if __name__ == '__main__':
    run_predict()