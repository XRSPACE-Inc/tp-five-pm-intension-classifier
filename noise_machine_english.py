import os
import random
import re
import argparse
from datetime import datetime

def apply_typo_rule(word):
    """
    根據 Typo 規則隨機插入、刪除、替換或交換字符
    """
    if not word:
        return word

    typo_type = random.choice(["insert", "delete", "replace", "swap"])
    if len(word) <= 1:
        typo_type = random.choice(["insert", "replace"])

    if typo_type == "insert":
        # 隨機位置插入一個字符
        pos = random.randint(0, len(word))
        char_to_insert = random.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:pos] + char_to_insert + word[pos:]

    elif typo_type == "delete" and len(word) > 1:
        # 隨機刪除一個字符
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos + 1:]

    elif typo_type == "replace":
        # 隨機替換一個字符
        pos = random.randint(0, len(word) - 1)
        char_to_replace = random.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:pos] + char_to_replace + word[pos + 1:]

    elif typo_type == "swap" and len(word) > 1:
        # 隨機交換兩個字符的位置
        idx1, idx2 = random.sample(range(len(word)), 2)
        text_list = list(word)
        text_list[idx1], text_list[idx2] = text_list[idx2], text_list[idx1]
        return ''.join(text_list)

    return word

def count_lines(file_path):
    """
    計算檔案內的行數
    """
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)
    
def generate_output_filename(output_dir):
    """
    自動生成唯一的輸出檔案名稱
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"output_{timestamp}.txt")

def process_text(input_file, output_dir, typo_probability, max_lines):
    """
    處理文字並隨機應用拼寫錯誤，持續新增直到檔案達到指定行數
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = generate_output_filename(output_dir)

    while count_lines(output_file) < max_lines:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = [line.strip() for line in infile if line.strip()]

        full_text = ' '.join(lines)
        sentences = [sentence.strip() + '.' for sentence in full_text.split('. ') if sentence.strip()]

        processed_lines = []

        for sentence in sentences:
            words = re.split(r'(\s+|[.,])', sentence)
            processed_words = []

            for word in words:
                if not word.strip():
                    continue
                if random.random() < typo_probability and re.fullmatch(r"[a-zA-ZàâäÉéèêëîïôöùûüç'-]+", word):
                    processed_words.append(apply_typo_rule(word))
                else:
                    processed_words.append(word)
            processed_words.append("\t")
            processed_words.append(sentence)
            processed_lines.append(" ".join(processed_words))

        processed_lines = [s.replace(" ,", ",").replace(" .", ".") for s in processed_lines]

        with open(output_file, 'a', encoding='utf-8') as outfile:
            outfile.write("\n".join(processed_lines) + "\n")

    print(f"Processing complete. Output file with added noise: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="產生包含拼寫錯誤的訓練資料")
    parser.add_argument("--input", type=str, required=True, help="輸入檔案名稱")
    parser.add_argument("--output", type=str, default="training_data", help="輸出檔案名稱")
    parser.add_argument("--typo_probability", type=float, default=0.15, help="拼寫錯誤的機率 (預設為 0.15)")
    parser.add_argument("--max_lines", type=int, default=1000, help="輸出檔案的最大行數 (預設為 1000)")

    args = parser.parse_args()

    process_text(args.input, args.output, args.typo_probability, args.max_lines)