import time

import csv
import os
import random
import jieba
import threading
import noise_machine_chinese

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from py_chinese_pronounce import Word2Pronounce, Pronounce2Word
from multiprocessing import current_process

Max_Sentence_Per_File = 1000

W2P = Word2Pronounce()
P2W = Pronounce2Word()

Wrong_Prob = 0.15


def is_chinese_char(c: str) -> bool:
    return '\u4e00' <= c <= '\u9fa5'


def is_english_letter(c: str) -> bool:
    return ord('A') <= ord(c) <= ord('Z') or ord('a') <= ord(c) <= ord('z')


def transform_letter(letter: str) -> str:
    if not letter.isalpha() or len(letter) != 1:
        raise ValueError("Input must be a single English letter.")

    if letter.isupper():
        # 如果字母是大寫，從 A-Z 中隨機選一個
        return random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    else:
        # 如果字母是小寫，從 a-z 中隨機選一個
        return random.choice('abcdefghijklmnopqrstuvwxyz')


def jieba_seg_(_text: str) -> list[str]:
    s_seg_pos = jieba.tokenize(_text)
    res = [seg for seg, _, _ in s_seg_pos]
    return res


def should_be_replaced() -> bool:
    wrong_prob = Wrong_Prob
    return random.random() < wrong_prob


def gen_sub_pair(ori_sentence: str, segments: list[str], generator, gen_num: int = 5) -> list[str]:
    result = []
    for _ in range(gen_num):
        wrong_sentence = ""
        for seg in segments:
            for idx, c in enumerate(seg):
                if should_be_replaced():
                    if is_chinese_char(c):
                        try:
                            r = confused_char_generator.get_replacement_char(text=seg, index=idx, generator=generator)
                        except Exception as e:
                            print(f"Error getting replacement char: {e}")
                            r = c
                        wrong_sentence += r
                    elif is_english_letter(c):
                        # r = transform_letter(c)
                        wrong_sentence += c
                    else:
                        wrong_sentence += c
                else:
                    wrong_sentence += c

        combined_string = '\t'.join([wrong_sentence, ori_sentence])
        result.append(combined_string)

    return result


def save_to_file(sentences, input_file_prefix, file_count):
    print(f"save_to_file, input_file_prefix={input_file_prefix}, file_count={file_count}")
    output_directory = r'D:\cht_data\training_data_ver0517_ccc_ver03'
    print(f"output_directory={output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    output_file_name = os.path.join(output_directory, f'{input_file_prefix}_output_{file_count}.txt')
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')


def generate_pair(file_name, max_sentences_per_file: int = 1000):
    print(f"任務 {file_name} 正在由 {threading.current_thread().name} 執行")
    print(f"任務 {file_name} 正在由 {current_process().name} 執行")
    generator = confused_char_generator.ConfusedCharGenerator()  # 在每个线程中创建一个新的 generator 实例
    results = []
    file_count = 1
    input_file_prefix = os.path.splitext(os.path.basename(file_name))[0]
    print(f"input_file_prefix={input_file_prefix}")

    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            segments = jieba_seg_(_text=line)
            result = gen_sub_pair(ori_sentence=line, segments=segments, generator=generator)
            results.extend(result)

            if len(results) >= max_sentences_per_file:
                save_to_file(results, input_file_prefix, file_count)
                file_count += 1
                results = []

    print(f"out results={results}")
    if results:
        save_to_file(results, input_file_prefix, file_count)


def _simple_run():
    directory_path = r"D:\cht_data\processed_2"
    file_names = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]

    print(f"file_names={file_names}")

    if len(file_names) == 0:
        print(f"No .txt files found in directory '{directory_path}'.")
        return

    for file_name in file_names:
        generate_pair(file_name=file_name)

    print("All files processed successfully.")


if __name__ == "__main__":
    start_time = time.time()
    _simple_run()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time={execution_time}")
