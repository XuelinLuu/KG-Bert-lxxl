import os
import transformers
class KGExamples:
    def __init__(self, guid, text_a, text_c, relation):
        self.guid = guid
        self.text_a = text_a
        self.text_c = text_c
        self.relation = relation


class KGFeatues:
    def __init__(self):
        pass

class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError

    @classmethod
    def _read_tsv(cls, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = []
            for line in f.readlines():
                line = line.rstrip("\n")
                lines.append(line)
        return lines

class KGProcessor(DataProcessor):
    """
    训练三元组之用
    """
    def __init__(self):
        self.relations = ["place_of_birth",
                          "nationality",
                          "religion",
                          "gender",
                          "children",
                          "parents",
                          "institution",
                          "spouse",
                          "place_of_death",
                          "cause_of_death",
                          "ethnicity",
                          "location",
                          "profession"]

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        entities2text = self._get_all_entitie2text(data_dir)
        return self._create_examples(lines, "train", entities2text)

    def _get_all_entitie2text(self, data_dir):
        entities2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                entities2text[line[0]] = line[1]
        return entities2text

    def _get_all_rela2text(self, data_dir):
        relations2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                relations2text[line[0]] = line[1]
        return relations2text

    def _get_all_text2rela(self, data_dir):
        text2relations = {}
        with open(os.path.join(data_dir, "relation2text.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                text2relations[line[1]] = line[0]
        return text2relations

    def get_all_relations(self):
        return self.relations

    def _create_examples(self, lines, type_str, entities2text):
        examples = []
        for idx, line in enumerate(lines):
            guid = f"{type_str}-{idx}"
            a, b, c = line.rstrip("\n").split("\t")
            text_a = entities2text[a]
            text_c = entities2text[c]

            examples.append(KGExamples(
                guid=guid,
                text_a=text_a,
                text_c=text_c,
                relation=b
            ))
        return examples

    def convert_examples_to_features(self, examples, max_seq_len, tokenizer):
        rela2ids = {relation: idx for idx, relation in enumerate(self.relations)}
        features = []
        for idx, example in enumerate(examples):
            head, tail, rela = example.text_a, example.text_c, example.relation
            head = tokenizer.tokenize(head)
            tail = tokenizer.tokenize(tail)
            len_head, len_tail = len(head), len(tail)
            while len_head + len_tail > max_seq_len - 3:
                if len_head > len_tail:
                    head = head[:len_head-1]
                else:
                    tail = tail[:len_tail-1]
                len_head = len(head)
                len_tail = len(tail)

            tokens = ["[CLS]"] + head + ["[SEP]"] + tail + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * (len(head) + 2) + [1] * (len(tail) + 1)

            if len(input_ids) < max_seq_len:
                padding_len = max_seq_len - len(input_ids)
                input_ids += [0] * padding_len
                attention_mask += [0] * padding_len
                token_type_ids += [0] * padding_len

            label = rela2ids[rela]
            features.append({
                "tokens": tokens,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "label": label
            })
        return features


class KGPredicProcessor():
    '''
    预测三元组
    '''
    def __init__(self):
        self.relations = ["place_of_birth",
                          "nationality",
                          "religion",
                          "gender",
                          "children",
                          "parents",
                          "institution",
                          "spouse",
                          "place_of_death",
                          "cause_of_death",
                          "ethnicity",
                          "location",
                          "profession"]

    def get_examples(self, data_dir):
        return self._create_examples(data_dir)


    def _get_all_triples(self, data_dir):
        triples = []
        with open(os.path.join(data_dir, "train.tsv"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                triples.append(f"\t{line[0]}\t{line[2]}")
        return triples

    def _get_all_entities(self, data_dir):
        entities = []
        with open(os.path.join(data_dir, "entities.txt"), "r", encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx > 100:
                    break
                line = line.rstrip("\n")
                entities.append(line)
        return entities

    def _get_all_entitie2text(self, data_dir):
        entities2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                entities2text[line[0]] = line[1]
        return entities2text

    def _get_all_text2entities(self, data_dir):
        text2entities = {}
        with open(os.path.join(data_dir, "entity2text.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                text2entities[line[1]] = line[0]
        return text2entities

    def get_all_relations(self):
        return self.relations

    def _create_examples(self, data_dir):
        all_triples = self._get_all_triples(data_dir)
        all_entities = self._get_all_entities(data_dir)
        examples = []
        entities2text = self._get_all_entitie2text(data_dir)
        for i, entity_i in enumerate(all_entities):
            for j, entity_j in enumerate(all_entities):
                if i == j:
                    continue
                if f"\t{entity_i}\t{entity_j}" in all_triples or f"\t{entity_j}\t{entity_i}" in all_triples:
                    continue
                examples.append({
                    "a": entity_i,
                    "b": entity_j,
                    "text_a": entities2text[entity_i],
                    "text_b": entities2text[entity_j]
                })
        return examples

    def convert_examples_to_features(self, examples, max_seq_len, tokenizer):
        features = []
        for idx, example in enumerate(examples):
            head, tail, text_head, text_tail = example["a"], example["b"], example["text_a"], example["text_b"]
            text_head = tokenizer.tokenize(text_head)
            text_tail = tokenizer.tokenize(text_tail)
            len_head, len_tail = len(text_head), len(text_tail)
            while len_head + len_tail > max_seq_len - 3:
                if len_head > len_tail:
                    text_head = text_head[:len_head-1]
                else:
                    text_tail = text_tail[:len_tail-1]
                len_head = len(text_head)
                len_tail = len(text_tail)

            tokens = ["[CLS]"] + text_head + ["[SEP]"] + text_tail + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * (len(text_head) + 2) + [1] * (len(text_tail) + 1)

            if len(input_ids) < max_seq_len:
                padding_len = max_seq_len - len(input_ids)
                input_ids += [0] * padding_len
                attention_mask += [0] * padding_len
                token_type_ids += [0] * padding_len

            features.append({
                "head_tail": f"{head}\t{tail}",
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            })
        return features




if __name__ == '__main__':
    kgp = KGProcessor()
    examples = kgp.get_train_examples("../datasets")
    tok = transformers.BertTokenizer.from_pretrained("../bert-base-uncased/vocab.txt")
    features = kgp.convert_examples_to_features(examples, 768, tok)
    print(examples[0].text_a)
    print(examples[0].text_c)
    print(examples[0].relation)
    print(features)