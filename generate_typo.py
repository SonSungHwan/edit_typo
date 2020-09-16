from __future__ import unicode_literals

import random
import numpy as np
import argparse
import re
from hangul_char_util import check_syllable, split_syllables, split_syllable_char, readlines_file, write_file

KEYBOARD_DICT = {'ㄲ': [('ㄸ', 'I'), ('ㅆ', 'IF')], 'ㅆ': [('ㄲ', 'IF')],
                 'ㄱ': [('ㄷ', 'IF'), ('ㅅ', 'IF')], 'ㅅ': [('ㄱ', 'IF')], 'ㅛ': [('ㅕ', 'M')],
                 'ㅕ': [('ㅛ', 'M'), ('ㅑ', 'M')], 'ㄹ': [('ㅇ', 'IF'), ('ㅎ', 'IF')],
                 'ㅎ': [('ㄹ', 'IF')], 'ㅗ': [('ㅓ', 'M')], 'ㅊ': [('ㅌ', 'IF'), ('ㅍ', 'IF')],
                 'ㅍ': [('ㅊ', 'IF')], 'ㅠ': [('ㅜ', 'M')], 'ㄺ': [('ㄽ', 'F')], 'ㄽ': [('ㄺ', 'F')],
                 'ㅃ': [('ㅉ', 'I')], 'ㅉ': [('ㅃ', 'I'), ('ㄸ', 'I')], 'ㄸ': [('ㅉ', 'I'), ('ㄲ', 'IF')],
                 'ㅒ': [('ㅑ', 'M'), ('ㅖ', 'M')], 'ㅖ': [('ㅒ', 'M')], 'ㅂ': [('ㅈ', 'IF')],
                 'ㅈ': [('ㅂ', 'IF'), ('ㄷ', 'IF')], 'ㄷ': [('ㅈ', 'IF'), ('ㄱ', 'IF')],
                 'ㅁ': [('ㄴ', 'IF')], 'ㄴ': [('ㅁ', 'IF'), ('ㅇ', 'IF')], 'ㅇ': [('ㄴ', 'IF'), ('ㄹ', 'IF')],
                 'ㅋ': [('ㅌ', 'IF')], 'ㅌ': [('ㅋ', 'IF'), ('ㅊ', 'IF')], 'ㅑ': [('ㅕ', 'M'), ('ㅐ', 'M')],
                 'ㅐ': [('ㅑ', 'M'), ('ㅔ', 'M')], 'ㅔ': [('ㅐ', 'M')], 'ㅓ': [('ㅗ', 'M'), ('ㅏ', 'M')],
                 'ㅏ': [('ㅓ', 'M'), ('ㅣ', 'M')], 'ㅣ': [('ㅏ', 'M')], 'ㅜ': [('ㅠ', 'M'), ('ㅡ', 'M')],
                 'ㅡ': [('ㅜ', 'M')], 'ㅘ': [('ㅚ', 'M')], 'ㅚ': [('ㅘ', 'M')], 'ㅟ': [('ㅢ', 'M')], 'ㅢ': [('ㅟ', 'M')]}

# shift_ja, mo, mid_key, side_key
# 핸드폰 QWERTY 키보드에서 shift가 필요한 자음과 모음, 중앙 키(엄지 손가락과 거리가 멈), 양 옆 키(엄지 손가락과 거리가 가까움)
# 양 옆 간섭 오타, 완성형 한글이 되는 경우의 오타만 허용
TOTAL_TYPO_TYPE = [set(['ㄲ', 'ㅆ', 'ㅃ', 'ㅉ', 'ㄸ']),
                   set(['ㅒ', 'ㅖ']),
                   set(['ㄱ', 'ㅅ', 'ㅛ', 'ㅕ', 'ㄹ', 'ㅎ', 'ㅗ', 'ㅊ', 'ㅍ', 'ㅠ', 'ㄺ', 'ㄽ']),
                   set(['ㅂ', 'ㅈ', 'ㄷ', 'ㅁ', 'ㄴ', 'ㅇ', 'ㅋ', 'ㅌ', 'ㅑ', 'ㅐ', 'ㅔ', 'ㅓ', 'ㅏ', 'ㅣ', 'ㅜ', 'ㅡ',
                    'ㅘ', 'ㅚ', 'ㅟ', 'ㅢ'])]

# 각 타입에 따른 오타 생성 비율
TYPO_TYPE_PRO_LIST = [0.25, 0.25, 0.25, 0.25]


def typo_type_probability(pro_list, n_iter):
    count = np.zeros(len(pro_list), dtype=int)

    for iter in range(n_iter):
        ratio = random.random()
        temp = 0
        for i, p in enumerate(pro_list):
            if ratio >= temp and ratio < (p + temp):
                count[i] += 1
            temp += p

    return count


def check_IMF(word, pro_char):
    char_list = []
    change_list = []
    num = 0

    for i, char in enumerate(word):
        temp = []
        c_temp = []
        if check_syllable(char):
            split_char = split_syllable_char(char)
            for c in split_char:
                if c in pro_char:
                    c_temp.append(1)  # 문제!
                    num += 1
                else:
                    c_temp.append(0)

            temp.append((split_char[0], 'INITIALS'))
            temp.append((split_char[1], 'MEDIALS'))
            if len(split_char) == 3:
                temp.append((split_char[2], 'FINALS'))
        else:
            temp.append((char, 'none'))
            c_temp.append(0)
        change_list.append(c_temp)
        char_list.append(temp)
    return char_list, change_list, num


def change_c(c_info):
    c, info = c_info
    candidate = KEYBOARD_DICT[c]
    choice_pro = 1 / len(candidate)

    while (True):
        probability = random.random()

        for cand in candidate:
            if probability < choice_pro:
                if info == 'INITIALS' and cand[1] in ['I', 'IF']:
                    return [(cand[0], info)]
                elif info == 'MEDIALS' and cand[1] == 'M':
                    return [(cand[0], info)]
                elif info == 'FINALS' and cand[1] in ['IF', 'F']:
                    return [(cand[0], info)]
                elif cand[1] == 'char':
                    temp = split_syllable_char(cand[0])
                    return [(temp[0], 'INITIALS'), (temp[1], 'MEDIALS')]
            probability = 1 - probability


def word_typo(word, pro_char):
    char_list, change_list, num = check_IMF(word, pro_char)
    check_pos = -1
    change_char_pos = random.randrange(num)

    for i, char in enumerate(change_list):
        for j, c in enumerate(char):
            check_pos += c
            if change_char_pos == check_pos:
                sub_char = change_c(char_list[i][j])
                if len(sub_char) > 1:
                    # print(char_list)
                    del char_list[i][j]
                    char_list = np.insert(char_list, i + 1, sub_char)
                    return char_list
                else:
                    char_list[i][j] = sub_char[0]
                    return char_list


def check_word_typo_type(words, pro_list):
    check_list = np.zeros((len(words), len(pro_list)), dtype=int)
    word_pos = np.array(([]), dtype=int)
    result_pro_list = np.zeros(len(pro_list), dtype=int)

    for i, word in enumerate(words):
        chars = set(list(split_syllables(word)))
        for n in range(len(pro_list)):
            type_set = set(TOTAL_TYPO_TYPE[n])
            if len(chars & type_set) > 0:
                check_list[i][n] += 1

    for i, pro in enumerate(pro_list):
        result_pro_list[i] = 0
        if pro > 0:
            type_list = np.where(check_list[:, i] == 1)[0]
            n = pro
            if pro >= len(type_list):  # 정해진 오타 개수 유지를 위해서는 수정해야함. 선정한 오타 타입의 오타 개수보다
                # 실제 해당 오타 타입에 해당하는 단어가 적은 경우 오타 개수를 유지시키는 문제
                word_pos = np.r_[word_pos, type_list]
                # 사실상 일정 수 이상의 어절로 구성된 문장을 사용
                result_pro_list[i] = len(type_list)
                check_list[type_list] = np.zeros(len(pro_list), dtype=int)
                pro_list[i + 1] += (pro - len(type_list))
            else:
                while (n > 0):
                    pos = random.randrange(len(type_list))
                    if type_list[pos] not in word_pos:
                        word_pos = np.append(word_pos, type_list[pos])
                        check_list[type_list[pos]] = np.zeros(
                            len(pro_list), dtype=int)
                        result_pro_list[i] += 1
                        n -= 1
    return word_pos, result_pro_list


def join_char_list(char_list):
    join = ''

    for chars in char_list:
        for c in chars:
            join += c[0]
    return join


def make_typo(sentence, typo_type_pro_list, n_typo):
    pro_list = typo_type_probability(typo_type_pro_list, n_typo)
    words = np.array((sentence.split()))

    typo_pos, result_type_list = check_word_typo_type(words, pro_list)
    typo_words = []

    type_num = 0
    for i, t in enumerate(result_type_list):
        if t > 0:
            for j in range(t):
                typo_words.append(join_char_list(
                    word_typo(words[typo_pos][type_num], TOTAL_TYPO_TYPE[i])))
                type_num += 1

    result_typo_sentence = []
    for word in words:
        result_typo_sentence.append(split_syllables(word))

    for i, pos in enumerate(typo_pos):
        result_typo_sentence[pos] = typo_words[i]

    return result_typo_sentence, typo_pos


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus_file", default='./KCC150_Korean_sentences_UTF8.txt', type=str, required=True, help="corpus data path"
    )
    parser.add_argument(
        "--output_path", default='./', type=str, reqrired=True, help="output result folder path"
    )
    parser.add_argument(
        "--per_sent_typo_num", default=3, type=int, reqrired=True, help="number of typos per sentence"
    )

    args = parser.parse_args()

    lines = readlines_file(args.corput_file)
    result = ''

    for i, line in enumerate(lines):
        sent_list, typo_pos = make_typo(
            line.strip(), TYPO_TYPE_PRO_LIST, args.per_sent_typo_num)
        typo_sent = ' '.join(sent_list)
        result += '<' + line.strip() + '\n'
        result += '>' + typo_sent + '\n'
        result += re.sub(r'[\[\]]', '', str(typo_pos)) + '\n'

    write_file(args.output_path + 'typo_text.txt', result.strip())


if __name__ == "__main__":
    main()
