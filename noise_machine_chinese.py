import csv
import os
import random
import time
import logging
import coloredlogs
from datetime import datetime
from dataclasses import dataclass

from py_chinese_pronounce import Word2Pronounce, Pronounce2Word


def _set_basic_logging_config():
    # 設定日誌的基本配置
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 最低接受的日誌等級

    # 創建文件日誌處理器，設置等級為 ERROR
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'log_{current_time}.txt'
    file_handler = logging.FileHandler(filename, 'w', 'utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 創建控制台日誌處理器，設置等級為 DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 添加處理器到日誌器
    logger.addHandler(file_handler)
    coloredlogs.install(level=logging.DEBUG, fmt='%(asctime)s - %(levelname)s - %(message)s', logger=logger)
    # logger.addHandler(console_handler)


COMMON_CHAR_LIST = []


def SET_COMMON_CHAR_LIST():
    global COMMON_CHAR_LIST
    common_chars = []
    cur_dir_path = os.path.dirname(__file__)
    file_path = os.path.join(cur_dir_path, "essential_data", "common_char.csv")
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            common_chars.extend(row)
    COMMON_CHAR_LIST = common_chars


SET_COMMON_CHAR_LIST()


@dataclass(frozen=True)
class FUZZY_LEVEL:
    SAME_HOMOPHONE: str = "same_homophone"
    DIFF_TONE: str = "diff_tone"
    CONFUSED_CHAR: str = "confused_char"
    RANDOM_COMMON_CHAR: str = "random_common_char"


class ConfusedCharGenerator:
    CONSONANT_LIST = [
        "ㄅㄆ", "ㄇㄈㄏ", "ㄉㄊ", "ㄋㄌㄖ", "ㄍㄎ",
        "ㄐㄑㄒ", "ㄓㄗ", "ㄔㄘ", "ㄕㄙ"
    ]
    VOWEL_LIST = [
        ["ㄜ", "ㄦ"], ["ㄧㄚ", "ㄧㄤ"], ["ㄧㄛ", "ㄧㄡ", "ㄩㄥ"],
        ["ㄞ", "ㄧㄞ"], ["ㄨㄞ", "ㄨㄢ"], ["ㄧ", "ㄧㄣ", "ㄧㄥ", "ㄩㄣ"],
        ["ㄠ", "ㄧㄠ"], ["ㄨㄚ", "ㄨㄤ"], ["ㄛ", "ㄡ", "ㄨㄛ", "ㄨㄥ"],
        ["ㄨ", "ㄩ"], ["ㄨㄟ", "ㄩㄝ"], ["ㄝ", "ㄟ", "ㄧㄝ", "ㄧㄢ", "ㄩㄢ"],
        ["ㄚ", "ㄢ", "ㄤ"], ["ㄣ", "ㄥ", "ㄨㄣ"]
    ]

    COMMON_CHARS = COMMON_CHAR_LIST

    def __init__(self):
        self.W2P = Word2Pronounce()
        self.P2W = Pronounce2Word()
        # self.COMMON_CHARS = []
        # self._set_common_chars()

    # def _set_common_chars(self):
    #     self.COMMON_CHARS = COMMON_CHAR_LIST

    def generate_confused_char(self, text, index, fuzzy_level, **kwargs):
        ori_ch = text[index]
        chewin = kwargs.get('chewin', None)

        if chewin is None:
            chewin = self.W2P.sent_to_chewin(text)[index]

        method_name = getattr(self, f'get_{fuzzy_level}', None)

        if not callable(method_name):
            raise AttributeError(f'Method get_{fuzzy_level} not found in {self.__class__.__name__}')

        return method_name(ori_ch, chewin)

    def get_same_homophone(self, ori_ch, c_chewin):
        new_char = ori_ch
        homophones = [hom for hom in self.P2W.chewin2word(c_chewin) if hom in self.COMMON_CHARS and hom != ori_ch]
        if homophones:
            r = random.choice(homophones)
            new_char = r

        logging.debug(f"new_char={new_char},ori_ch={ori_ch},homophones={homophones},c_chewin={c_chewin}")
        return new_char

    def get_diff_tone(self, ori_ch, c_chewin):
        new_char = ori_ch
        tones = "ˊˇˋ˙"
        base_chewin = c_chewin[:-1] if c_chewin[-1] in tones else c_chewin
        tone_candidates = [base_chewin + tone for tone in tones if base_chewin + tone != c_chewin]
        if base_chewin != c_chewin:
            tone_candidates += [base_chewin]

        candidates = []
        for tone_chewin in tone_candidates:
            diff_tone_words = self.P2W.chewin2word(tone_chewin)
            logging.debug(f"tone_chewin={tone_chewin}, diff_tone_words={diff_tone_words}")
            if diff_tone_words:
                candidates += diff_tone_words

        logging.debug(f"candidates={candidates}")
        candidates = [c for c in candidates if c in self.COMMON_CHARS and c != ori_ch]
        if candidates:
            r = random.choice(candidates)
            new_char = r

        return new_char

    def get_confused_char(self, ori_ch, c_chewin):
        new_char = ori_ch
        tones = "ˊˇˋ˙"
        c_tone = c_chewin[-1] if c_chewin[-1] in tones else ''
        base_chewin = c_chewin[:-1] if c_chewin[-1] in tones else c_chewin
        if not base_chewin:
            print(f"Empty base_chewin detected for ori_ch={ori_ch}, c_chewin={c_chewin}")
            return new_char

        consonant_candidates = ""
        c_consonant = ""
        found_consonant = False
        for consonants in self.CONSONANT_LIST:
            for consonant in consonants:
                if base_chewin[0] == consonant:
                    c_consonant = consonant
                    consonant_candidates = [v for v in consonants if v != c_consonant]
                    base_chewin = base_chewin[1:]
                    found_consonant = True
                    break

            if found_consonant:
                break

        vowel_candidates = []
        c_vowel = ""
        for vowels in self.VOWEL_LIST:
            for vowel in vowels:
                if base_chewin == vowel:
                    c_vowel = vowel
                    vowel_candidates = [v for v in vowels if v != c_vowel]
                    break

        possible_pronounces = []
        for p_consonant in consonant_candidates:
            p_chewin = p_consonant + c_vowel + c_tone
            p_chars = self.P2W.chewin2word(p_chewin)
            possible_pronounces.extend(p_chars)

        for p_vowel in vowel_candidates:
            p_chewin = c_consonant + p_vowel + c_tone
            p_chars = self.P2W.chewin2word(p_chewin)
            possible_pronounces.extend(p_chars)

        possible_pronounces = [c for c in possible_pronounces if c in self.COMMON_CHARS and c != ori_ch]
        logging.debug(f"possible_pronounces={possible_pronounces}")
        if possible_pronounces:
            r = random.choice(possible_pronounces)
            new_char = r

        return new_char

    def get_random_common_char(self, ori_ch, chewin):
        if self.COMMON_CHARS:
            return random.choice(self.COMMON_CHARS)

        logging.warning(f"COMMON_CHARS not found")
        return ori_ch


def get_replacement_char(text: str, index: int, generator=None) -> str:
    """
    word + index = char -> replaced char
    """

    if generator is None:
        generator = ConfusedCharGenerator()

    probability_distribution = [
        (0.4, FUZZY_LEVEL.SAME_HOMOPHONE),
        (0.25, FUZZY_LEVEL.DIFF_TONE),
        (0.25, FUZZY_LEVEL.CONFUSED_CHAR),
        (0.1, FUZZY_LEVEL.RANDOM_COMMON_CHAR)
    ]

    random_value = random.random()
    cumulative_probability = 0.0
    selected_fuzzy_level = None
    for probability, fuzzy_level in probability_distribution:
        cumulative_probability += probability
        if random_value < cumulative_probability:
            selected_fuzzy_level = fuzzy_level
            break

    replacement_char = generator.generate_confused_char(text, index, selected_fuzzy_level)
    return replacement_char


def _simple_test():
    generator = ConfusedCharGenerator()
    text = "行人"

    fuzzy_levels = [FUZZY_LEVEL.SAME_HOMOPHONE, FUZZY_LEVEL.DIFF_TONE, FUZZY_LEVEL.CONFUSED_CHAR,
                    FUZZY_LEVEL.RANDOM_COMMON_CHAR]

    for count in range(5):
        logging.debug(f"count={count}")
        for index in range(len(text)):
            logging.debug(f"要測的字={text}的{text[index]}")
            for fuzzy_level in fuzzy_levels:
                _start_time = time.perf_counter()
                res = generator.generate_confused_char(text, index, fuzzy_level)
                logging.debug(f"fuzzy_level={fuzzy_level}, res={res}")
                _end_time = time.perf_counter()
                execution_time = _end_time - _start_time
                logging.debug(f"execution_time={execution_time} second")


if __name__ == '__main__':
    _set_basic_logging_config()
    _simple_test()
