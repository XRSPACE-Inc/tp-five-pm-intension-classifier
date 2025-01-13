import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoModel


# Define your dataset class
class SpellingCorrectionDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []  # List of (input, target) tuples
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                source_text, target_text = line.strip().split("\t")
                self.data.append((source_text, target_text))

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text, target_text = self.data[idx]
        # 编码输入和目标文本，并确保所有的输出具有相同的长度
        processed_input = self.tokenizer(source_text, add_special_tokens=True, max_length=128, padding="max_length",
                                         truncation=True)
        processed_target = self.tokenizer(target_text, add_special_tokens=True, max_length=128,
                                          padding="max_length",
                                          truncation=True)

        input_ids = torch.tensor(processed_input['input_ids'])
        target_ids = torch.tensor(processed_target['input_ids'])

        return input_ids, target_ids


# Define the fine-tuned ALBERT model for spelling correction
class AlbertForSpellingCorrection(nn.Module):
    def __init__(self, albert_model, tokenizer):
        super(AlbertForSpellingCorrection, self).__init__()
        self.albert = albert_model  # ALBERT model for masked language modeling
        self.fc = nn.Linear(albert_model.config.hidden_size, tokenizer.vocab_size)

    def forward(self, input_ids):
        outputs = self.albert(input_ids=input_ids)
        logits = self.fc(outputs.last_hidden_state)
        return logits


def test_albert(model, tokenizer, device, source_text):
    model.eval()
    input_ids = tokenizer.encode(source_text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(input_ids)

    predicted_ids = torch.argmax(logits, dim=-1).squeeze().tolist()  # Get the most likely token IDs
    predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)

    return predicted_text, predicted_ids


def _training_script():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    albert_model = AutoModel.from_pretrained('ckiplab/albert-tiny-chinese')

    # Define paths for your training and validation data
    train_data_path = r"D:\csc_task_datset\ver_01\training_data.txt"
    valid_data_path = r"D:\csc_task_datset\validation_data.txt"

    # Create datasets for training and validation
    train_dataset = SpellingCorrectionDataset(data_path=train_data_path, tokenizer=tokenizer)
    valid_dataset = SpellingCorrectionDataset(data_path=valid_data_path, tokenizer=tokenizer)

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)  # No need to shuffle validation data

    # Initialize model
    model = AlbertForSpellingCorrection(albert_model, tokenizer)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for input_ids, target_ids in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)

            train_loss = loss_fn(logits, target_ids)

            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

        # Validation phase
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for input_ids, target_ids in tqdm(valid_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Valid)"):
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)

                logits = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                target_ids = target_ids.view(-1)

                valid_loss = loss_fn(logits, target_ids)

                total_valid_loss += valid_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        # avg_valid_loss = total_valid_loss / len(valid_dataloader)

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")
            current_time = datetime.datetime.now()
            timestamp_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(model.state_dict(), f"spelling_correction_model_{timestamp_string}.pth")

    # Save the trained model
    current_time = datetime.datetime.now()
    timestamp_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(model.state_dict(), f"spelling_correction_model_{timestamp_string}.pth")


if __name__ == '__main__':

    start_time = time.time()
    _training_script()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time={execution_time}")
