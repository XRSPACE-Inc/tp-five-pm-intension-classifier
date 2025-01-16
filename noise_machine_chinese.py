import time
import csv
import os
import random
import jieba
import threading
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from py_chinese_pronounce import Word2Pronounce, Pronounce2Word
from multiprocessing import current_process


Max_Sentence_Per_File = 1000  # 將由 argparse 解析的參數覆蓋
W2P = Word2Pronounce()
P2W = Pronounce2Word()
Wrong_Prob = 0.15

COMMON_CHAR_LIST: List[str] = []
TONE_MARKS = "\u02ca\u02c7\u02cb\u02d9"


def set_common_char_list():
    """加載 common_char.csv 內的常用字到全域變數 COMMON_CHAR_LIST"""
    global COMMON_CHAR_LIST
    cur_dir_path = os.path.dirname(__file__)
    file_path = os.path.join(cur_dir_path, "essential_data", "common_char.csv")
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            COMMON_CHAR_LIST = [char for row in csv_reader for char in row]
    except FileNotFoundError as e:
        print(f"Common character file not found: {e}")
        raise
    except Exception as e:
        print(f"Error reading common character file: {e}")
        raise


set_common_char_list()


@dataclass(frozen=True)
class FuzzyLevel:
    SAME_HOMOPHONE: str = "same_homophone"
    DIFF_TONE: str = "diff_tone"
    CONFUSED_CHAR: str = "confused_char"
    RANDOM_COMMON_CHAR: str = "random_common_char"


class ConfusedCharGenerator:
    """生成與原字形相近或同音的混淆字"""

    CONSONANT_LIST = [
        "\u3105\u3106", "\u3107\u3108\u310f", "\u3109\u310a", "\u310b\u310c\u3117", "\u310d\u310e",
        "\u3110\u3111\u3112", "\u3113\u3114", "\u3115\u3116", "\u3118\u3119"
    ]
    VOWEL_LIST = [
        ["\u311d", "\u3126"], ["\u3127\u311a", "\u3127\u3125"], ["\u3127\u311b", "\u3127\u3129", "\u3128\u3129"],
        ["\u311e", "\u3127\u311e"], ["\u3128\u311e", "\u3128\u3124"], ["\u3127", "\u3127\u3123", "\u3127\u3125", "\u3128\u3123"],
        ["\u311f", "\u3127\u311f"], ["\u3128\u311a", "\u3128\u3125"], ["\u311b", "\u3129", "\u3128\u311b", "\u3128\u3129"],
        ["\u3128", "\u3129"], ["\u3128\u3122", "\u3128\u3127"], ["\u311e", "\u311f", "\u3127\u311e", "\u3127\u3124"],
        ["\u311a", "\u3124", "\u3125"], ["\u3123", "\u3125", "\u3128\u3123"]
    ]

    def __init__(self):
        self.W2P = Word2Pronounce()
        self.P2W = Pronounce2Word()

    def generate_confused_char(self, text: str, index: int, fuzzy_level: str, **kwargs) -> str:
        """根據模糊級別生成混淆字"""
        ori_ch = text[index]
        chewin = kwargs.get('chewin') or self.W2P.sent_to_chewin(text)[index]
        method = getattr(self, f'get_{fuzzy_level}', None)
        if not callable(method):
            raise AttributeError(f"Method get_{fuzzy_level} not found in {self.__class__.__name__}")
        return method(ori_ch, chewin)

    def get_same_homophone(self, ori_ch: str, chewin: str) -> str:
        """生成同音字（不同字形）"""
        homophones = [hom for hom in self.P2W.chewin2word(chewin) if hom in COMMON_CHAR_LIST and hom != ori_ch]
        return random.choice(homophones) if homophones else ori_ch

    def get_diff_tone(self, ori_ch: str, chewin: str) -> str:
        """生成不同聲調的同音字"""
        base_chewin = chewin[:-1] if chewin[-1] in TONE_MARKS else chewin
        tone_candidates = [base_chewin + tone for tone in TONE_MARKS if base_chewin + tone != chewin]
        if base_chewin != chewin:
            tone_candidates.append(base_chewin)

        candidates = [word for tone in tone_candidates for word in self.P2W.chewin2word(tone)
                      if word in COMMON_CHAR_LIST and word != ori_ch]
        return random.choice(candidates) if candidates else ori_ch

    def get_confused_char(self, ori_ch: str, chewin: str) -> str:
        """生成聲母或韻母不同但易混淆的字"""
        return ori_ch

    def get_random_common_char(self, ori_ch: str, chewin: str) -> str:
        """隨機返回一個常用字"""
        return random.choice(COMMON_CHAR_LIST) if COMMON_CHAR_LIST else ori_ch


def get_replacement_char(text: str, index: int, generator: Optional[ConfusedCharGenerator] = None) -> str:
    generator = generator or ConfusedCharGenerator()
    probability_distribution = [
        (0.4, FuzzyLevel.SAME_HOMOPHONE),
        (0.25, FuzzyLevel.DIFF_TONE),
        (0.25, FuzzyLevel.CONFUSED_CHAR),
        (0.1, FuzzyLevel.RANDOM_COMMON_CHAR)
    ]

    random_value = random.random()
    cumulative_probability = 0.0
    selected_fuzzy_level = None
    for probability, fuzzy_level in probability_distribution:
        cumulative_probability += probability
        if random_value < cumulative_probability:
            selected_fuzzy_level = fuzzy_level
            break

    return generator.generate_confused_char(text, index, selected_fuzzy_level)


def jieba_seg_(_text: str) -> list[str]:
    s_seg_pos = jieba.tokenize(_text)
    return [seg for seg, _, _ in s_seg_pos]


def should_be_replaced() -> bool:
    return random.random() < Wrong_Prob


def gen_sub_pair(ori_sentence: str, segments: list[str], generator, gen_num: int = 5) -> list[str]:
    result = []
    for _ in range(gen_num):
        wrong_sentence = ""
        for seg in segments:
            for idx, c in enumerate(seg):
                if should_be_replaced():
                    if '一' <= c <= '龥':
                        try:
                            r = get_replacement_char(text=seg, index=idx, generator=generator)
                        except Exception as e:
                            print(f"Error getting replacement char: {e}")
                            r = c
                        wrong_sentence += r
                    else:
                        wrong_sentence += c
                else:
                    wrong_sentence += c

        combined_string = '\t'.join([wrong_sentence, ori_sentence])
        result.append(combined_string)

    return result


def save_to_file(sentences, input_file_prefix, file_count, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    output_file_name = os.path.join(output_directory, f'{input_file_prefix}_output_{file_count}.txt')
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')


def generate_pair(file_name, max_sentences_per_file, output_directory):
    generator = ConfusedCharGenerator()
    results = []
    file_count = 1
    input_file_prefix = os.path.splitext(os.path.basename(file_name))[0]

    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            segments = jieba_seg_(_text=line)
            result = gen_sub_pair(ori_sentence=line, segments=segments, generator=generator)
            results.extend(result)

            if len(results) >= max_sentences_per_file:
                save_to_file(results, input_file_prefix, file_count, output_directory)
                file_count += 1
                results = []

    if results:
        save_to_file(results, input_file_prefix, file_count, output_directory)


def process_text(directory_path, max_sentences_per_file, output_directory):
    file_names = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]

    if not file_names:
        print(f"No .txt files found in directory '{directory_path}'.")
        return

    for file_name in file_names:
        generate_pair(file_name, max_sentences_per_file, output_directory)

    print("All files processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate incorrect sentence pairs.")
    parser.add_argument("--directory_path", type=str, required=True, help="Path to the input directory containing .txt files.")
    parser.add_argument("--max_sentences_per_file", type=int, default=1000, help="Maximum number of sentences per output file.")
    parser.add_argument("--output_directory", type=str, default='chinese_output_data', help="Path to the output directory.")

    args = parser.parse_args()

    start_time = time.time()
    process_text(args.directory_path, args.max_sentences_per_file, args.output_directory)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")
