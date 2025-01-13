import os
import random
import re
import argparse


def replace_keyboard(word, keyboard_neighbors):
    """
    隨機將單詞中的字母替換為鍵盤鄰近的字母
    """
    if not word:
        return word

    new_word = list(word)
    for i, char in enumerate(new_word):
        lower_char = char.lower()
        if lower_char in keyboard_neighbors:
            replacement = random.choice(keyboard_neighbors[lower_char])
            new_word[i] = replacement.upper() if char.isupper() else replacement

    return ''.join(new_word)


def contains_umlaut(word, umlaut_mapping):
    """
    檢查單詞中是否包含重音字符
    """
    return any(char in umlaut_mapping for char in word)


def replace_umlaut(word, umlaut_mapping):
    """
    將單詞中的重音字符替換為對應的無重音字符
    """
    for umlaut, replacement in umlaut_mapping.items():
        word = word.replace(umlaut, replacement)
    return word


def apply_typo_rule(word):
    """
    根據不同規則隨機插入、刪除、替換字符，或替換重音字符
    """
    if not word:
        return word

    umlaut_mapping = {'ä': 'a', 'ö': 'o', 'ü': 'u', 'Ä': 'A', 'Ö': 'O', 'Ü': 'U'}
    keyboard_neighbors = {
        'a': 'sqw', 'b': 'vng', 'c': 'xdfv', 'd': 'erfcx', 'e': 'rdsw',
        'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'uojk', 'j': 'uikmnh',
        'k': 'iolmj', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'ipl',
        'p': 'ol', 'q': 'wa', 'r': 'tfde', 's': 'awedx', 't': 'rgyf',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qsae', 'x': 'sdcz', 'y': 'xzut',
        'z': 'ayx'
    }

    typo_type = random.choice(["insert", "delete", "replace", "swap", "umlaut"])

    if contains_umlaut(word, umlaut_mapping):
        if len(word) <= 1:
            typo_type = random.choice(["insert", "replace", "umlaut"])

    if typo_type == "insert":
        pos = random.randint(0, len(word))
        char_to_insert = random.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:pos] + char_to_insert + word[pos:]

    elif typo_type == "delete" and len(word) > 1:
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos + 1:]

    elif typo_type == "replace":
        return replace_keyboard(word, keyboard_neighbors)

    elif typo_type == "swap" and len(word) > 1:
        idx1, idx2 = random.sample(range(len(word)), 2)
        word_list = list(word)
        word_list[idx1], word_list[idx2] = word_list[idx2], word_list[idx1]
        return ''.join(word_list)

    elif typo_type == "umlaut":
        return replace_umlaut(word, umlaut_mapping)

    return word


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
    accent_pattern = r"[a-zA-ZäöüÄÖÜß'-]+"

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
