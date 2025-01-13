import csv
import os
import random
import time
import re

# def replace_with_typo(content, accent_mapping, replacement_prob=0.15):
#     # words = content.split()
#     # for i, word in enumerate(words):
#     #     if random.random() < replacement_prob and word in typo_rules:
#     #         words[i] = typo_rules[word]
#     # return ' '.join(words)
#     new_text = []
#     for char in content:
#         if char in accent_mapping and random.random() < replacement_prob:
#             new_text.append(accent_mapping[char])
#         else:
#             new_text.append(char)
#     print(new_text)
#     return ''.join(new_text)
#
# def process_Accents_Errors(input_file, output_file, replacement_prob=0.15):
#     # Define typo rules based on keyboard layout errors
#     typo_rules = {
#         "congé": "conge",
#         "à": "a",
#         "é": "e",
#         "ç": "c",
#         "été": "ete",
#         "garçon": "garcon",
#     }
#     accent_mapping = {
#         'é': 'e',
#         'è': 'e',
#         'ê': 'e',
#         'ë': 'e',
#         'à': 'a',
#         'ù': 'u',
#         'â': 'a',
#         'î': 'i',
#         'ô': 'o',
#         'û': 'u',
#         'ç': 'c'
#     }
#     with open(input_file, 'r', encoding='utf-8') as infile:
#         content = infile.read()
#
#     typo_content = replace_with_typo(content, typo_rules, replacement_prob)
#
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         outfile.write(typo_content)

def replace_accents_noise(content, accent_mapping):
    new_text = []
    for char in content:
        if char in accent_mapping:
            new_text.append(accent_mapping[char])
        else:
            new_text.append(char)
    return ''.join(new_text)


def contains_accent(word, accent_mapping):
    # 檢查 word 中是否有任何字符在 accent_mapping 中
    for char in word:
        if char in accent_mapping:
            return True
    return False

def apply_typo_rule(word):
    """根據 Typo 規則隨機插入、刪除或替換字符"""
    if not word:
        return word

    accent_mapping = {
        'é': 'e',
        'è': 'e',
        'ê': 'e',
        'ë': 'e',
        'à': 'a',
        'ù': 'u',
        'â': 'a',
        'î': 'i',
        'ô': 'o',
        'û': 'u',
        'ç': 'c'
    }

    typo_type = random.choice(["insert", "delete", "replace", "accents", "swap"])

    if contains_accent(word, accent_mapping):
        typo_type = random.choice(["insert", "delete", "replace", "accents", "swap"])
        if len(word) <= 1:
            typo_type = random.choice(["insert", "replace", "accents"])
            print(typo_type)
    else:
        typo_type = random.choice(["insert", "delete", "replace", "swap"])
        if len(word) <= 1:
            typo_type = random.choice(["insert", "replace"])

    if typo_type == "insert":
        # 隨機位置插入一個字符
        pos = random.randint(0, len(word))
        char_to_insert = random.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:pos] + char_to_insert + word[pos:]

    elif typo_type == "delete":
        # 隨機刪除一個字符
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos + 1:]

    elif typo_type == "replace":
        # 隨機替換一個字符
        pos = random.randint(0, len(word) - 1)
        char_to_replace = random.choice("abcdefghijklmnopqrstuvwxyz")
        return word[:pos] + char_to_replace + word[pos + 1:]

    elif typo_type == "accents":
        return replace_accents_noise(word, accent_mapping)

    elif typo_type == "swap":
        letter_indices = [i for i, char in enumerate(word) if char.isalpha()]
        idx1, idx2 = random.sample(letter_indices, 2)
        text_list = list(word)
        text_list[idx1], text_list[idx2] = text_list[idx2], text_list[idx1]
        return ''.join(text_list)

    return word

# def process_insert_delete_replace(input_file, output_file, typo_probability=0.15):
#     """處理文字並隨機應用拼寫錯誤"""
#     with open(input_file, 'r', encoding='utf-8') as infile:
#         lines = infile.readlines()
#
#     processed_lines = []
#
#     for line in lines:
#         words = line.split()
#         processed_words = []
#
#         for word in words:
#             if random.random() < typo_probability:
#                 processed_words.append(apply_typo_rule(word))
#             else:
#                 processed_words.append(word)
#
#         processed_lines.append(" ".join(processed_words))
#
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         outfile.write("\n".join(processed_lines))
##################################################################
def replace_with_homophones(word, homophones_dict):
    """替換為同音異義詞，如果該詞在同音詞字典中。"""
    return homophones_dict.get(word, word)

def apply_homophone_errors(text, homophones_dict, error_rate=0.15):
    """根據錯誤率替換文字中的單詞為同音異義詞。"""
    words = text.split()
    total_words = len(words)
    num_errors = int(total_words * error_rate)

    # 隨機選擇要替換的單詞索引
    error_indices = random.sample(range(total_words), num_errors)

    for idx in error_indices:
        words[idx] = replace_with_homophones(words[idx], homophones_dict)

    return ' '.join(words)

def is_letter(char):
    return char.isalpha()

# def process_homophones(input_file, output_file, typo_probability=0.15):
#     # 同音異義詞字典
#     homophones_dict = {
#         "ver": "vert",
#         "vert": "ver",
#         "sait": "c'est",
#         "c'est": "sait",
#         "son": "sont",
#         "sont": "son",
#         "et": "est",
#         "est": "et"
#     }
#     with open(input_file, "r", encoding="utf-8") as f:
#         input_text = f.read()
#
#     # 應用同音異義詞錯誤
#     output_text = apply_homophone_errors(input_text, homophones_dict)
#
#     # 將結果寫入輸出檔案
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write(output_text)
##################################################################
def process_text(input_file, output_file, typo_probability=0.15):
    # 同音異義詞字典
    homophones_dict = {
        "ver": "vert",
        "vert": "ver",
        "sait": "c'est",
        "c'est": "sait",
        "son": "sont",
        "sont": "son",
        "et": "est",
        "est": "et"
    }

    # Define typo rules based on keyboard layout errors
    # typo_rules = {
    #     "congé": "conge",
    #     "à": "a",
    #     "é": "e",
    #     "ç": "c",
    #     "été": "ete",
    #     "garçon": "garcon",
    # }
    # accent_mapping = {
    #     'é': 'e',
    #     'è': 'e',
    #     'ê': 'e',
    #     'ë': 'e',
    #     'à': 'a',
    #     'ù': 'u',
    #     'â': 'a',
    #     'î': 'i',
    #     'ô': 'o',
    #     'û': 'u',
    #     'ç': 'c'
    # }
    new_text = []

    # with open(input_file, 'r', encoding='utf-8') as infile:
    #     content = infile.read()
    # typo_content = replace_with_typo(content, accent_mapping)

    # with open(input_file, "r", encoding="utf-8") as f:
    #     input_text = f.read()
    # # 應用同音異義詞錯誤
    # output_text = apply_homophone_errors(input_text, homophones_dict)
    # # 將結果寫入輸出檔案
    # with open(output_file, "w", encoding="utf-8") as f:
    #     f.write(output_text)

    ##################################################################
    """處理文字並隨機應用拼寫錯誤"""
    with open(input_file, 'r', encoding='utf-8') as infile:
        # 逐行讀取，刪除空行和首尾空白
        lines = [line.strip() for line in infile if line.strip()]
        # lines = infile.readlines()
    # 將所有行合併成一個完整的字串
    full_text = ' '.join(lines)
    # 使用句號切割字串，並去除每個句子的首尾空白
    # sentences = [sentence.strip() for sentence in full_text.split('.') if sentence.strip()]
    sentences = [sentence.strip() + '.' for sentence in full_text.split('. ') if sentence.strip()]

    processed_lines = []

    # for line in lines:
    for sentence in sentences:
        # words = sentence.split()
        words = re.split(r'(\s+|[.,])', sentence)
        processed_words = []

        for word in words:
            if not word.strip():
                continue
            # if random.random() < typo_probability and is_letter(word):
            if random.random() < typo_probability and re.fullmatch(r"[a-zA-ZàâäÉéèêëîïôöùûüç'-]+", word):
                processed_words.append(apply_typo_rule(word))
            else:
                processed_words.append(word)
        processed_words.append("\t")
        processed_words.append(sentence)
        processed_lines.append(" ".join(processed_words))
        processed_lines = [s.replace(" ,", ",").replace(" .", ".") for s in processed_lines]

    # question = input("Question: ")

    # with open(output_file, 'w', encoding='utf-8') as outfile:
    #     for sentence in sentences:
    #         outfile.write(sentence + '\n')
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("\n".join(processed_lines))
    ##################################################################

    # with open(input_file, 'r', encoding='utf-8') as infile:
    #     content = infile.read()
    # print(content)
    # question = input("Question: ")
    #
    # typo_content = replace_with_typo(content, typo_rules, replacement_prob)
    #
    # with open(output_file, 'w', encoding='utf-8') as outfile:
    #     outfile.write(typo_content)
##################################################################

if __name__ == "__main__":
    input_file = "input_test.txt"  # 輸入檔案名稱
    output_file = "output_test.txt"  # 輸出檔案名稱


    # Process the file
    process_text(input_file, output_file)
    # process_homophones(input_file, output_file)
    # process_Accents_Errors(input_file, output_file)
    # process_insert_delete_replace(input_file, output_file)













# def apply_typo_rules(word):
#     """依據四種規則隨機替換單字"""
#     rules = [
#         "keyboard_layout_error",
#         "spelling_typo",
#         "neighboring_key_error",
#         "homophone_confusion"
#     ]
#     selected_rule = random.choice(rules)
#
#     if selected_rule == "keyboard_layout_error":
#         replacements = {
#             "congé": "conge",
#             "à": "a"
#         }
#         return replacements.get(word, word)
#
#     elif selected_rule == "spelling_typo":
#         replacements = {
#             "la vue": "lavue",
#             "résultat": "resulat"
#         }
#         return replacements.get(word, word)
#
#     elif selected_rule == "neighboring_key_error":
#         replacements = {
#             "bonjour": "bonjoru",
#             "avec": "qvec"
#         }
#         return replacements.get(word, word)
#
#     elif selected_rule == "homophone_confusion":
#         replacements = {
#             "vert": "ver",
#             "c'est": "sait"
#         }
#         return replacements.get(word, word)
#
#     return word
#
# def process_homophones(file_path, output_path):
#     """處理輸入的文字檔案並生成包含錯誤的文字檔案"""
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#
#     processed_lines = []
#
#     for line in lines:
#         words = re.findall(r'\w+|\W+', line)  # 分離文字與標點符號
#         new_words = []
#
#         for word in words:
#             if word.isalpha() and random.random() < 0.15:  # 15% 機率替換
#                 new_words.append(apply_typo_rules(word))
#             else:
#                 new_words.append(word)
#
#         processed_lines.append(''.join(new_words))
#
#     with open(output_path, 'w', encoding='utf-8') as output_file:
#         output_file.writelines(processed_lines)
#
# if __name__ == "__main__":
#     input_file = "input.txt"  # 替換成您的輸入檔案名稱
#     output_file = "output.txt"  # 輸出檔案名稱
#     process_homophones(input_file, output_file)
