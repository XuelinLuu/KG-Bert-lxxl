import tqdm
import torch
import torch.nn as nn
def loss_fn(outputs, labels):
    loss = nn.BCEWithLogitsLoss()
    return loss(outputs, labels)

def train_fn(model, device, train_dataloader, optimizer, lr_scheduler=None):
    model.train()
    tk = tqdm.tqdm(train_dataloader, desc="Train Iter")
    train_loss = 0
    for idx, data in enumerate(tk):
        input_ids, attention_mask, token_type_ids = data[0], data[1], data[2]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = data[3].to(device)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        train_loss += loss.item()
        avg_loss = train_loss / (idx+1)
        tk.set_postfix(idx=idx, loss=loss.item(), avg_loss=avg_loss)


def prdict_fn(model, device, train_dataloader, rela_list, head_tails, prediction_output_path, threshold):
    model.eval()
    tk = tqdm.tqdm(train_dataloader, desc="Predict Iter")
    append_num = 0
    with open(prediction_output_path, "a", encoding="utf-8") as f:
        with torch.no_grad():
            for idx, data in enumerate(tk):
                input_ids, attention_mask, token_type_ids = data[0], data[1], data[2]

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                preds = torch.argmax(outputs, dim=-1)
                preds = preds.squeeze()
                outputs = outputs.detach().cpu().numpy()

                if outputs[0][preds.item()] > threshold:
                    append_num += 1
                    rela = rela_list[preds.item()]
                    head_tail = head_tails[idx]
                    head, tail = head_tail.split("\t")
                    f.write(f"{head}\t{rela}\t{tail}\n")

    print(append_num)



