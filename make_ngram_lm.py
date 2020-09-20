# make ngram data code lib
import numpy as np
import re
import argparse
import logging
from collections import Counter
from collections import defaultdict
from tqdm import tqdm

from hangul_char_util import readlines_file, read_file, save_json, check_syllable, check_han_word

NUMBER = '<<NUM>>'
ENG = '<<ENG>>'
SENT_START = '<<SOS>>'
SENT_END = '<<EOS>>'


def n_gram(chars, n):
    return list(zip(*[chars[i:] for i in range(n)]))


def get_word_list(text):  # word 단위 리스트 생성
    return re.split(r'[\n ]+', text)


def syllable_add_SEtoken(word, n):  # 음절 ngram 정제, word 시작, 끝 토큰 추가
    # word = re.sub('[a-zA-Z]+', eng, word) #영어 단어를 하나의 토큰으로 정함
    replace_num_word = re.sub('\d+', NUMBER, word)

    result = []
    temp = ''
    for c in replace_num_word:
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

    for word in sent.split():
        word = re.sub('\d+', NUMBER, word)
        words.append(word)

    for _ in range(n_gram - 1):
        words.append(SENT_END)  # 역방향 ngram 구할 때 사용
        words.insert(0, SENT_START)
    return words


def fwbw_n_gram(n_gram, syl_ngram):
    forward = []
    backward = []

    for n in n_gram:
        if syl_ngram:
            if len(n[-1]) == 1:
                if check_syllable(n[-1]):
                    forward.append(n)
            if len(n[0]) == 1:
                if check_syllable(n[0]):
                    backward.append(n)
        else:
            if check_han_word(n[-1]):
                forward.append(n)
            if check_han_word(n[0]):
                backward.append(n)

    return forward, backward


def preprocessing(lines, n, syl_ngram):
    p_forward = []
    b_forward = []

    if not syl_ngram:
        for line in lines:
            p_f_words, p_b_words = fwbw_n_gram(
                n_gram(get_sent_list(line.strip(), n), n), syl_ngram)
            if len(p_f_words) > 0:
                p_forward.append(p_f_words)
            if len(p_b_words) > 0:
                b_forward.append(p_b_words)
    else:
        words = get_word_list('\n'.join(lines))

        for word in words:
            p_f_word, p_b_word = fwbw_n_gram(
                n_gram(syllable_add_SEtoken(word, n), n), syl_ngram)
            if len(p_f_word) > 0:
                p_forward.append(p_f_word)
            if len(p_b_word) > 0:
                b_forward.append(p_b_word)

    return p_forward, b_forward


def calc_n_gram(text, n, f_calc_result, b_calc_result, syl_ngram, last):
    join_str = '' if syl_ngram else ' '
    forward, backward = preprocessing(text, n, syl_ngram)
    forward = np.array(forward)
    backward = np.array(backward)

    for i, f_sw in enumerate(forward):
        b_sw = backward[i]
        for j, f_wc in enumerate(f_sw):
            f_calc_result[join_str.join(f_wc[:-1])][f_wc[-1]] += 1
            b_wc = b_sw[j]
            b_calc_result[join_str.join(b_wc[1:])][b_wc[0]] += 1

    if last:
        for key in f_calc_result.keys():
            total = sum(f_calc_result[key].values())
            for c in f_calc_result[key]:
                f_calc_result[key][c] = f_calc_result[key][c] / total
                #, f_calc_result[key][c]] # [확률, 빈도수] 형식

        for key in b_calc_result.keys():
            total = sum(b_calc_result[key].values())
            for c in b_calc_result[key]:
                b_calc_result[key][c] = b_calc_result[key][c] / total
                #, b_calc_result[key][c]]

    return f_calc_result, b_calc_result


def mass_data_lm(path, line_num, n_gram, syl_ngram):
    lines = readlines_file(path)
    len_lines = len(lines)
    f_calc_result = defaultdict(Counter)
    b_calc_result = defaultdict(Counter)
    index = 0

    while (1):
        if (index + line_num) > len_lines:
            f_calc_result, b_calc_result = calc_n_gram(
                lines[index:], n_gram, f_calc_result, b_calc_result, syl_ngram, True)
            break
        else:
            f_calc_result, b_calc_result = calc_n_gram(lines[index: index + line_num], n_gram, f_calc_result,
                                                       b_calc_result, syl_ngram, False)
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

    return {k: cnt[k] for k in cnt if cnt[k] > 1}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus_file", default='./KCC150_Korean_sentences_UTF8.txt', type=str, required=True, help="corpus data path"
    )
    parser.add_argument(
        "--output_path", default='./', type=str, required=True, help="output result folder path"
    )
    parser.add_argument(
        "--ngram_n", default=2, type=int, required=True, help="set n in ngram"
    )
    parser.add_argument(
        "--per_process_line_num", default=100000, type=int, help="per process sentence num"
    )
    parser.add_argument(
        "--syl_ngram", action='store_true', help="if True make syllable ngram"
    )

    args = parser.parse_args()

    if args.ngram_n == 1:
        save_json(args.output_path + 'unigram.json', unigram(args.corpus_file))
    elif args.ngram_n > 1:
        f_result, b_result = mass_data_lm(
            args.corpus_file, args.per_process_line_num, args.ngram_n, args.syl_ngram)
        if args.syl_ngram:
            save_json(f'{args.output_path}{args.ngram_n}gram_fc_result.json', f_result)
            save_json(f'{args.output_path}{args.ngram_n}gram_bc_result.json', b_result)
        else:
            save_json(f'{args.output_path}{args.ngram_n}gram_fw_result.json', f_result)
            save_json(f'{args.output_path}{args.ngram_n}gram_bw_result.json', b_result)


if __name__ == "__main__":
    main()
