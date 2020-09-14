#make ngram data code lib
import numpy as np
import re
import argparse
import logging
from collections import Counter
from tqdm import tqdm

from hangul_char_util import readlines_file, save_json, check_syllable


NUMBER = '<<NUM>>'
ENG = '<<ENG>>'
SENT_START = '<<SOS>>'
SENT_END = '<<EOS>>'

corpus_path = 'C:/Users/ssh/Desktop/작업/data/KCC150_Korean_sentences_UTF8/KCC150_Korean_sentences_UTF8.txt'

def n_gram(chars, n):
    return list(zip(*[chars[i:] for i in range(n)]))


def get_word_list(text):  # word 단위 리스트 생성
    return re.split(r'[\n ]+', text)


def syllable_add_SEtoken(word, n):  # 음절 ngram 정제, word 시작, 끝 토큰 추가
    # word = re.sub('[a-zA-Z]+', eng, word) #영어 단어를 하나의 토큰으로 정함
    word = re.sub('\d+', NUMBER, word)

    result = []
    temp = ''
    for c in word:
        if check_syllable(c):
            if len(temp) > 0:
                result.append(temp)
                temp = ''
            result.append(c)
        else:
            temp += c
            if temp == NUMBER:
                result.append(temp)
                temp = ''
            elif re.search('[a-zA-Z]+<<NUM>>', temp) != None:
                temp = temp.replace(NUMBER, '')
                result.append(temp)
                result.append(NUMBER)
                temp = ''
    if len(temp) > 0:
        result.append(temp)

    for i in range(n - 1):
        result.append(SENT_END)
        result.insert(0, SENT_START)

    return result

def get_sent_list(sent, n_gram):  # sent 단위 리스트 생성
    words = []
    sent = sent.split()

    for word in sent:
        word = re.sub('\d+', NUMBER, word)
        words.append(word)

    for i in range(n_gram - 1):
        words.append(SENT_END)  # 역방향 ngram 구할 때 사용
        words.insert(0, SENT_START)
    return words

def fwbw_n_gram(n_gram):
    forward = []
    backward = []

    for chars in n_gram:
        if len(chars[-1]) == 1:
            if check_syllable(chars[-1]):
                forward.append(chars)
        if len(chars[0]) == 1:
            if check_syllable(chars[0]):
                backward.append(chars)

    return forward, backward

def preprocessing(lines, n, word_ngram= True):
    p_forward = []
    b_forward = []

    if word_ngram:
        for line in lines:
            p_f_words, p_b_words = fwbw_n_gram(n_gram(get_sent_list(line, n), n))
            if len(p_f_words) > 0:
                p_forward.append(p_f_words)
            if len(p_b_words) > 0:
                b_forward.append(p_b_words)
    else:
        words = get_word_list('\n'.join(lines.strip()))

        for word in words:
            p_f_word, p_b_word = fwbw_n_gram(n_gram(syllable_add_SEtoken(word, n), n))
            if len(p_f_word) > 0:
                p_forward.append(p_f_word)
            if len(p_b_word) > 0:
                b_forward.append(p_b_word)

    return p_forward, b_forward


def calc_n_gram(text, n, f_calc_result, b_calc_result, last):
    # f_calc_result = {}
    # b_calc_result = {}

    forward, backward = preprocessing(text, n)
    forward = np.array(forward)
    backward = np.array(backward)

    for i, f_word in enumerate(forward):
        b_word = backward[i]
        for j, f_chars in enumerate(f_word):
            if ''.join(f_chars[:-1]) not in f_calc_result:
                f_calc_result[''.join(f_chars[:-1])] = Counter()
            if f_calc_result[''.join(f_chars[:-1])][f_chars[-1]] == 0:
                f_calc_result[''.join(f_chars[:-1])][f_chars[-1]] = 1
            else:
                f_calc_result[''.join(f_chars[:-1])][f_chars[-1]] += 1

            b_chars = b_word[j]
            if ''.join(b_chars[1:]) not in b_calc_result:
                b_calc_result[''.join(b_chars[1:])] = Counter()
            if b_calc_result[''.join(b_chars[1:])][b_chars[0]] == 0:
                b_calc_result[''.join(b_chars[1:])][b_chars[0]] = 1
            else:
                b_calc_result[''.join(b_chars[1:])][b_chars[0]] += 1

    if last:  # 수정: [확률, 빈도수] 형식으로 바꿈
        for key in f_calc_result.keys():
            total = sum(f_calc_result[key].values())
            for c in f_calc_result[key]:
                f_calc_result[key][c] = [f_calc_result[key][c] / total, f_calc_result[key][c]]

        for key in b_calc_result.keys():
            total = sum(b_calc_result[key].values())
            for c in b_calc_result[key]:
                b_calc_result[key][c] = [b_calc_result[key][c] / total, b_calc_result[key][c]]

    return f_calc_result, b_calc_result


def mass_data_lm(path, line_num, n_gram):
    lines = readlines_file(path)
    len_lines = len(lines)

    f_calc_result = {}
    b_calc_result = {}
    index = 0

    while (1):
        if (index + line_num) > len_lines:
            f_calc_result, b_calc_result = calc_n_gram(lines[index:], n_gram, f_calc_result, b_calc_result, True)
            break
        else:
            f_calc_result, b_calc_result = calc_n_gram(lines[index: index + line_num], n_gram, f_calc_result,
                                                       b_calc_result, False)
            index += line_num

        if index % 100000 == 0:
            print(index)

    return f_calc_result, b_calc_result


def unigram(path):
    lines = readlines_file(path)
    cnt = Counter()

    for i, line in enumerate(tqdm(lines)):
        words = get_sent_list(line, 1)
        for word in words:
            cnt[word] += 1
        if i != 0 and i % 200000 == 0:
            print(i)

    return {k: cnt[k] for k in cnt if cnt[k] > 1}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus_file", default='C:/Users/ssh/Desktop/작업/data/KCC150_Korean_sentences_UTF8/KCC150_Korean_sentences_UTF8.txt', type=str, required=True, help="corpus data path"
    )
    parser.add_argument(
        "--output_path", default='./', type=str, reqrired=True, help="output result folder path"
    )
    parser.add_argument(
        "--ngram_n", default=2, type=int,  reqrired=True, help="set n in ngram"
    )
    parser.add_argument(
        "--per_process_line_num", default=2, type=int,  reqrired=True, help="set n in ngram"
    )

    args = parser.parse_args()

    if args.ngram_n == 1:
        save_json(args.output_path + 'unigram.json', unigram(args.corpus_file))
    elif args.ngram_n > 1:
        fw_result, bw_result= mass_data_lm(args.corpus_file, args.per_process_line_num, args.ngram_n)
        save_json(args.output_path + str(args.ngram_n)+'gram_fw_result.json', fw_result)
        save_json(args.output_path + str(args.ngram_n) + 'gram_bw_result.json', bw_result)

if __name__ == "__main__":
    main()