import datetime
import time
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse


def get_txt_files_in_directory(directory):
    # Initialize an empty list to store the paths of the .txt files
    txt_files = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_files.append(os.path.join(directory, filename))

    return txt_files


class SpellingCorrectionDataset(Dataset):
    def __init__(self, data_paths, tokenizer):
        self.data = []

        for data_path in data_paths:
            with open(data_path, "r", encoding="utf-8") as file:
                for line in file:
                    source_text, target_text = line.strip().split("\t")
                    self.data.append((source_text, target_text))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text, target_text = self.data[idx]
        processed_input = self.tokenizer(source_text,
                                         add_special_tokens=True,
                                         max_length=128,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors='pt')
        processed_target = self.tokenizer(target_text,
                                          add_special_tokens=True,
                                          max_length=128,
                                          padding="max_length",
                                          truncation=True,
                                          return_tensors='pt')

        input_ids = processed_input['input_ids'].squeeze()
        attention_mask = processed_input['attention_mask'].squeeze()
        target_ids = processed_target['input_ids'].squeeze()

        return input_ids, attention_mask, target_ids


class BertForSpellingCorrection(nn.Module):
    def __init__(self, bert_model, tokenizer):
        super(BertForSpellingCorrection, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(bert_model.config.hidden_size, tokenizer.vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state)
        return logits


def _training_script(directory, num_epochs, lr, save_interval):
    torch.cuda.empty_cache()

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    bert_model = BertModel.from_pretrained("bert-base-multilingual-uncased")

    train_data_path = get_txt_files_in_directory(directory)
    print(f"train_data_path={train_data_path}")

    valid_data_path = []

    train_dataset = SpellingCorrectionDataset(train_data_path, tokenizer)
    valid_dataset = SpellingCorrectionDataset(valid_data_path, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    model = BertForSpellingCorrection(bert_model, tokenizer)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for input_ids, attention_mask, target_ids in tqdm(train_dataloader,
                                                          desc=f"Epoch {epoch + 1}/{num_epochs} (Train)"):
            input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(
                device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)

            train_loss = loss_fn(logits, target_ids)
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        if (epoch + 1) % save_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")
            current_time = datetime.datetime.now()
            timestamp_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            model_name = f"typo_correction_model_epoch{epoch + 1}_{timestamp_string}.pth"
            torch.save(model.state_dict(), model_name)
            print(f"Model saved: {model_name}")

        torch.cuda.empty_cache()

    timestamp_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_model_name = f"typo_correction_model_final_{timestamp_string}.pth"
    torch.save(model.state_dict(), final_model_name)
    print(f"Final model saved: {final_model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Spelling correction model training script")
    parser.add_argument("--directory", type=str, required=True, help="訓練資料的目錄")
    parser.add_argument("--num_epochs", type=int, default=100, help="訓練的總 epoch 數 (預設為 100)")
    parser.add_argument("--lr", type=float, default=1e-5, help="學習率 (預設為 1e-5)")
    parser.add_argument("--save_interval", type=int, default=5, help="每隔多少個 epoch 儲存一次模型 (預設為 5)")

    args = parser.parse_args()
    start_time = time.time()
    _training_script(args.directory, args.num_epochs, args.lr, args.save_interval)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time={execution_time}")
