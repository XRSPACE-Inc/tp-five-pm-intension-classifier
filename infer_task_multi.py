import torch
import time
import argparse
from multi_version_training import BertForSpellingCorrection
from transformers import BertTokenizer, BertModel


def load_model(model_path):
    """Load the pre-trained BERT model and tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    bert_model = BertModel.from_pretrained("bert-base-multilingual-uncased")
    model = BertForSpellingCorrection(bert_model, tokenizer)

    # Load the saved model state dictionary
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict, strict=True)

    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


def correct_spelling(model, tokenizer, device, source_text):
    """Correct spelling errors in the source text using the model."""
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

    # Remove spaces
    predicted_text = predicted_text.replace(' ', '')
    return predicted_text


def main():
    """Main function to run the spelling correction interactively."""
    parser = argparse.ArgumentParser(description="BERT Spelling Correction Tool")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model file.")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path)

    while True:
        source_text = input("Please enter text (type 'exit' to quit): ")
        if source_text.lower() == 'exit':
            break

        start_time = time.time()
        predicted_text = correct_spelling(model, tokenizer, device, source_text)
        end_time = time.time()

        print(f"Source Text: {source_text}")
        print(f"Predicted Text: {predicted_text}")
        print(f"Inference execution time: {end_time - start_time:.4f}s\n")


if __name__ == "__main__":
    main()
