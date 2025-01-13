import torch
import time

from multi_version_training import BertForSpellingCorrection
from transformers import BertTokenizer, BertModel

# print(torch.cuda.is_available())
# print(torch.__version__)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bert_model = BertModel.from_pretrained("bert-base-multilingual-uncased")

# Initialize model
model = BertForSpellingCorrection(bert_model, tokenizer)

# Load the saved model state dictionary
model_path = "spelling_correction_model_2024-05-22_07-01-22.pth"
model_state_dict = torch.load(model_path)

# Load the complete state dictionary into the model
model.load_state_dict(model_state_dict, strict=True)

# Ensure model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# Define the testing function
def test(model, tokenizer, device, source_text):
    # Encode the source text, but now we keep the attention mask too
    encoding = tokenizer.encode_plus(
        source_text,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",  # make sure to specify padding
        truncation=True,  # enable truncation to max length
        max_length=128  # specify max length according to your needs or model config
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)  # Get attention mask

    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # Pass both input_ids and attention_mask

    predicted_ids = torch.argmax(logits, dim=-1).squeeze().tolist()  # Most likely token IDs
    predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)

    return predicted_text, predicted_ids


def correct_spelling(model, tokenizer, device, source_text):
    encoding = tokenizer.encode_plus(
        source_text,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    predicted_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
    predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)

    # 去掉空格
    predicted_text = predicted_text.replace(' ', '')
    return predicted_text, predicted_ids


# 測試新的函數
while True:
    print("Please enter text:")
    source_text = input()
    if source_text.lower() == 'exit':
        break
    start_time = time.time()

    predicted_text, predicted_ids = correct_spelling(model, tokenizer, device, source_text)

    print(f"Source Text: {source_text}")
    print(f"Predicted Text: {predicted_text}")

    # print(f"Predicted Token IDs: {predicted_ids}")
    # print(f"predicted_text={predicted_text},len(predicted_text)={len(predicted_text)}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Inference execution time: {execution_time}s")
