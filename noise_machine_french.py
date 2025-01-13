import os
import random
import re
import argparse


def replace_accents_noise(content, accent_mapping):
    """
    根據 accent_mapping 替換字串中的重音字符
    """
    return ''.join([accent_mapping.get(char, char) for char in content])


def contains_accent(word, accent_mapping):
    """
    檢查 word 中是否包含任何重音字符
    """
    return any(char in accent_mapping for char in word)


def apply_typo_rule(word):
    """
    根據不同規則隨機插入、刪除、替換字符，或替換重音字符
    """
    if not word:
        return word

    accent_mapping = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'ù': 'u', 'â': 'a', 'î': 'i',
        'ô': 'o', 'û': 'u', 'ç': 'c'
    }

    typo_type = random.choice(["insert", "delete", "replace", "accents", "swap"])

    # 若單詞包含重音字符，增加 "accents" 的選擇概率
    if contains_accent(word, accent_mapping):
        if len(word) <= 1:
            typo_type = random.choice(["insert", "replace", "accents"])
    else:
        if len(word) <= 1:
            typo_type = random.choice(["insert", "replace"])

    if typo_type == "insert":
        pos = random.randint(0, len(word))
        char_to_insert = random.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:pos] + char_to_insert + word[pos:]

    elif typo_type == "delete":
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos + 1:]

    elif typo_type == "replace":
        pos = random.randint(0, len(word) - 1)
        char_to_replace = random.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:pos] + char_to_replace + word[pos + 1:]

    elif typo_type == "accents":
        return replace_accents_noise(word, accent_mapping)

    elif typo_type == "swap" and len(word) > 1:
        idx1, idx2 = random.sample(range(len(word)), 2)
        text_list = list(word)
        text_list[idx1], text_list[idx2] = text_list[idx2], text_list[idx1]
        return ''.join(text_list)

    return word


def replace_with_homophones(word, homophones_dict):
    """
    若單詞存在於同音異義詞字典中，則替換為對應的同音異義詞
    """
    return homophones_dict.get(word, word)


def apply_homophone_errors(text, homophones_dict, error_rate=0.15):
    """
    根據錯誤率替換文字中的單詞為同音異義詞
    """
    words = text.split()
    num_errors = int(len(words) * error_rate)
    error_indices = random.sample(range(len(words)), num_errors)

    for idx in error_indices:
        words[idx] = replace_with_homophones(words[idx], homophones_dict)

    return ' '.join(words)


def count_lines(file_path):
    """
    計算檔案內的行數
    """
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)


def process_text(input_file, output_file, typo_probability, max_lines):
    """
    處理文字並隨機應用拼寫錯誤，持續新增直到檔案達到指定行數
    """
    accent_pattern = r"[a-zA-Zàâäéèêëîïôöùûüç'-]+"

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
                if random.random() < typo_probability and re.fullmatch(accent_pattern, word):
                    processed_words.append(apply_typo_rule(word))
                else:
                    processed_words.append(word)

            processed_lines.append(" ".join(processed_words).replace(" ,", ",").replace(" .", "."))

        with open(output_file, 'a', encoding='utf-8') as outfile:
            outfile.write("\n".join(processed_lines) + "\n")

    print(f"Processing complete. Output file with added noise: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="產生包含拼寫錯誤的訓練資料")
    parser.add_argument("--input", type=str, required=True, help="輸入檔案名稱")
    parser.add_argument("--output", type=str, default="test/output_test.txt", help="輸出檔案名稱")
    parser.add_argument("--typo_probability", type=float, default=0.15, help="拼寫錯誤的機率 (預設為 0.15)")
    parser.add_argument("--max_lines", type=int, default=1000, help="輸出檔案的最大行數 (預設為 1000)")

    args = parser.parse_args()

    process_text(args.input, args.output, args.typo_probability, args.max_lines)