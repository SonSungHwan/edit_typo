# edit_typo code lib

import re
import statistics
import time
import argparse
import logging
from itertools import product

from hangul_char_util import load_json, split_syllable_char, join_jamos, check_syllable, check_han_word
from generate_typo import change_c
from make_ngram_lm import syllable_add_SEtoken, n_gram

NUMBER = '<<NUM>>'
ENG = '<<ENG>>'
SENT_START = '<<SOS>>'
SENT_END = '<<EOS>>'

TOTAL_CHAR = set(['ㄲ', 'ㅆ', 'ㄱ', 'ㅅ', 'ㅛ', 'ㅕ', 'ㄹ', 'ㅎ', 'ㅗ', 'ㅊ', 'ㅍ', 'ㅠ', 'ㄺ', 'ㄽ', 'ㅃ', 'ㅉ',
                  'ㄸ', 'ㅒ', 'ㅖ', 'ㅂ', 'ㅈ', 'ㄷ', 'ㅁ', 'ㄴ', 'ㅇ', 'ㅋ', 'ㅌ', 'ㅑ', 'ㅐ', 'ㅔ', 'ㅓ', 'ㅏ',
                  'ㅣ', 'ㅜ', 'ㅡ', 'ㅘ', 'ㅚ', 'ㅟ', 'ㅢ'])


def make_candidats(char):
    cs = ''.join(split_syllable_char(char))
    candidats = [char]

    for i, c in enumerate(cs):
        if c in TOTAL_CHAR:
            temp = list(cs)
            if i == 0:
                cands = change_c(c, 'INITIALS')
                for cand in cands:
                    temp[0] = cand[0]
                    candidats.append(join_jamos(''.join(temp)))
            elif i == 1:
                cands = change_c(c, 'MEDIALS')
                for cand in cands:
                    temp[1] = cand[0]
                    candidats.append(join_jamos(''.join(temp)))
            else:
                cands = change_c(c, 'FINALS')
                for cand in cands:
                    temp[2] = cand[0]
                    candidats.append(join_jamos(''.join(temp)))

    return candidats


def word_candidats(word, unigram):  # 입력 어절의 후보 어절들의 유니그램 존재 여부에 따라 후보 어절을 걸러내는 모듈
    char_list = []
    word_candidate_list = []
    num_eng = ''
    origin_word = []

    for c in word:
        if check_syllable(c):  # 연속되는 숫자 및 영어 문자열을 하나의 토큰으로 리스트에 추가하기 위한 작업
            if num_eng != '':
                char_list.append([num_eng])
                origin_word.append(num_eng)
                num_eng = ''
            char_list.append(make_candidats(c))
            origin_word.append(c)
        else:
            num_eng += c

    if num_eng != '':
        char_list.append([num_eng])
        origin_word.append(num_eng)

    temp = origin_word[:]
    # 간섭오타로 어절의 한 글자씩 교체하여 후보 어절을 생성(unigram에 없는 단어라면 제외)
    for i, char_candidate in enumerate(char_list):
        if len(char_candidate) > 1:
            for c in char_candidate:
                temp[i] = c
                w_buff = ''.join(temp)
                num = 0
                if w_buff in unigram:
                    num = unigram[w_buff]
                if (w_buff, num) not in word_candidate_list:
                    word_candidate_list.append((w_buff, num))  # (후보 어절, 빈도수)
            temp = origin_word[:]

    return word_candidate_list


def syllable_edit_word(word, unigram, syl_tri_f, syl_tri_b, n_gram):  # 음절 ngram을 이용한 어절 수정 모듈
    chars = syllable_add_SEtoken(word.strip(), n_gram)

    score = 1.
    for i, char in enumerate(chars):
        if len(char) == 1:
            if check_syllable(char):
                f_chars = ''.join(chars[i - (n_gram - 1): i])
                b_chars = ''.join(chars[i + 1: i + n_gram])

                f_p, b_p = 0., 0.

                if f_chars in syl_tri_f.keys():
                    if char in syl_tri_f[f_chars].keys():
                        f_p = syl_tri_f[f_chars][char]

                if b_chars in syl_tri_b.keys():
                    if char in syl_tri_b[b_chars].keys():
                        b_p = syl_tri_b[b_chars][char]

                if f_p == 0 and b_p == 0:
                    score = 0.
                    break

                if i == 0:
                    score *= b_p
                else:
                    score *= ((f_p + b_p) / 2)

    if score > 0:
        return chars
    else:
        wc_list = word_candidats(word, unigram)
        wc_list[0] = (word, 0.)

    for i, wc in enumerate(wc_list[1:]):
        chars = syllable_add_SEtoken(wc[0], n_gram)
        score = 1.
        for j, char in enumerate(chars):
            if len(char) == 1:
                if check_syllable(char):
                    f_chars = ''.join(chars[j - (n_gram - 1): j])
                    b_chars = ''.join(chars[j + 1: j + n_gram])

                    f_p, b_p = 0., 0.
                    if f_chars in syl_tri_f.keys():
                        if char in syl_tri_f[f_chars].keys():
                            f_p = syl_tri_f[f_chars][char]

                    if b_chars in syl_tri_b.keys():
                        if char in syl_tri_b[b_chars].keys():
                            b_p = syl_tri_b[b_chars][char]

                    if j == 2:
                        score *= b_p
                    else:
                        score *= ((f_p + b_p) / 2)

                    if score == 0:
                        wc_list[i + 1] = (wc_list[i + 1][0], score)
                        break
        wc_list[i + 1] = (wc_list[i + 1][0], score)

    wc_list = sorted(wc_list, key=lambda x: x[1], reverse=True)
    if wc_list[0][1] == 0.:
        return chars
    else:
        return syllable_add_SEtoken(wc_list[0][0], n_gram)


def make_word_candidats(word, unigram, syl_tri_f, syl_tri_b, min_freq,
                        using_syl_edit):  # 입력 문장의 한 어절 단위로 후보 어절들을 생성하되 해당 어절의 후보 어절들이 전부 유니그램에 존재하지 않으면, 음절 단위로 수정
    candidate = []
    num_buff = []

    if re.search('\d+', word):  # 어절 ngram 검사를 위한 숫자 특수기호화 사전 작업
        num_buff = list(filter(None, re.split('\D+', word)))
        replace_num_word = re.sub('\d+', NUMBER, word)

    result = word_candidats(replace_num_word, unigram)  # 유니그램 체크를 통한 후보 어절 결과

    for r in result:
        if r[1] > min_freq:
            if len(num_buff) > 0:
                candidate_word = r[0]
                for n in num_buff:
                    candidate_word = candidate_word.replace(NUMBER, n, 1)
                candidate.append(candidate_word)
            else:
                candidate.append(r[0])

    if len(candidate) > 0:
        return candidate

    # 음절 n-gram을 사용하지 않는 경우를 확인할 때 수정 필요, 음절 ngram을 사용한 어절 수정 부분(후보 어절이 없다면, 원본 어절을 음절 ngram으로 수정)
    elif using_syl_edit:
        if len(num_buff) > 0:  # 수정해야 한다면 숫자도 복원
            num_restore_candi = result[0][0]
            for n in num_buff:
                num_restore_candi = num_restore_candi.replace(NUMBER, n, 1)
            candidate = [''.join(syllable_edit_word(
                num_restore_candi, unigram, syl_tri_f, syl_tri_b, 3)[2:-2])]
            candidate_word = candidate[0]
            for n in num_buff:
                candidate_word = candidate_word.replace(NUMBER, n, 1)
            # 음절 양방향 ngram 사용 어휘를 알기 위한 조치
            candidate = [candidate_word]
        else:
            candidate = [''.join(syllable_edit_word(
                result[0][0], unigram, syl_tri_f, syl_tri_b, 3)[2:-2])]

        if candidate[0] != result[0][0]:
            if len(num_buff) > 0:
                candidate.append(num_restore_candi)
            else:
                candidate.append(result[0][0])
        return candidate

    # 음절 ngram을 사용하지 않는 모듈
    elif len(num_buff) > 0 and not using_syl_edit:
        candidate_word = result[0][0]
        for n in num_buff:
            candidate_word = candidate_word.replace(NUMBER, n, 1)
        candidate = [candidate_word]
    else:
        candidate = [result[0][0]]

    return candidate


def make_sent_combi_list(sent, unigram, syl_tri_f, syl_tri_b, min_freq,
                         using_syl_edit):  # 문장을 어절 단위로 후보 어절들을 생성(원본은 유지 if 원본밖에 없다면, 후보 어절들의 Unigram 빈도수가 0인 경우)
    sent_combi_list = []

    for word in sent.split():
        if check_han_word(word):
            sent_combi_list.append(make_word_candidats(
                word, unigram, syl_tri_f, syl_tri_b, min_freq, using_syl_edit))
        else:
            sent_combi_list.append([word])

    sent_combi_list.insert(0, [SENT_START])
    sent_combi_list.append([SENT_END])
    return sent_combi_list


def edit_sent(sent, unigram, min_freq, word_bi_f, word_bi_b, word_tri_f, word_tri_b, syl_tri_f, syl_tri_b,
              using_syl_edit):
    edited_sent = []

    candidats_word_sent = make_sent_combi_list(
        sent, unigram, syl_tri_f, syl_tri_b, min_freq, using_syl_edit)
    # print(candidats_word_sent)
    tri_gram_candidats = n_gram(candidats_word_sent, 3)
    candidats_word_sent.insert(0, [SENT_START])
    candidats_word_sent.append([SENT_END])
    fif_gram_candidats = n_gram(candidats_word_sent, 5)

    for i, candidate_word in enumerate(candidats_word_sent[2:-2]):

        if len(candidate_word) > 1:  # 후보 어절이 없다면 그대로 유지(음절 ngram으로 수정 되거나 원본)
            select_list = {}
            # 목표 어절과 양 사이드 2개, 총 5개의 어절을 통해 어절 ngram을 활용하기 위한 사전 작업
            fif = list(product(*fif_gram_candidats[i]))
            for f in fif:
                target = re.sub('\d+', NUMBER, f[2])
                p_target = re.sub('\d+', NUMBER, f[1])
                pp_target = re.sub('\d+', NUMBER, f[0])
                a_target = re.sub('\d+', NUMBER, f[3])
                aa_target = re.sub('\d+', NUMBER, f[4])

                try:
                    word_tri_p = word_tri_f[pp_target + ' ' + p_target][target] / sum(
                        word_tri_f[pp_target + ' ' + p_target].values())
                except KeyError:
                    word_tri_p = 0.

                try:
                    word_tri_r_p = word_tri_b[a_target + ' ' + aa_target][target] / sum(
                        word_tri_b[a_target + ' ' + aa_target].values())
                except KeyError:
                    word_tri_r_p = 0.

                if f[2] not in select_list:
                    if i == 0 or i == len(candidats_word_sent[2:-2]) - 1:
                        if i == 0:
                            select_list[f[2]] = word_tri_r_p
                        else:
                            select_list[f[2]] = word_tri_p
                    else:
                        select_list[f[2]] = statistics.mean(
                            [word_tri_p, word_tri_r_p])
                else:
                    # if select_list[t[1]] < statistics.mean([word_bi_p, word_bi_r_p]):
                    if i == 0 or i == len(candidats_word_sent[2:-2]) - 1:
                        if i == 0:
                            select_list[f[2]] += word_tri_r_p
                        else:
                            select_list[f[2]] += word_tri_p
                    else:
                        select_list[f[2]
                                    ] += statistics.mean([word_tri_p, word_tri_r_p])

            sort_dict_t = sorted(select_list.items(), key=(
                lambda x: x[1]), reverse=True)
            if sort_dict_t[0][1] > 0:
                edited_sent.append(sort_dict_t[0][0])
                continue

            # elif len(candidate_word) > 1 and sort_dict_t[0][1] == 0.:
            select_list = {}
            tri = list(product(*tri_gram_candidats[i]))
            for t in tri:  # 우선적으로 target, pre_target, after_target 의 특수기호 추출
                target = re.sub('\d+', NUMBER, t[1])
                pre_target = re.sub('\d+', NUMBER, t[0])
                after_target = re.sub('\d+', NUMBER, t[2])

                try:
                    word_bi_p = word_bi_f[pre_target][target]
                except KeyError:
                    word_bi_p = 0.

                try:
                    word_bi_r_p = word_bi_b[after_target][target]
                except KeyError:
                    word_bi_r_p = 0.

                if t[1] not in select_list:
                    select_list[t[1]] = statistics.mean(
                        [word_bi_p, word_bi_r_p])
                else:
                    # if select_list[t[1]] < statistics.mean([word_bi_p, word_bi_r_p]):
                    select_list[t[1]
                                ] += statistics.mean([word_bi_p, word_bi_r_p])

            sort_dict = sorted(select_list.items(), key=(
                lambda x: x[1]), reverse=True)
            # print(sort_dict)
            if sort_dict[0][1] > 0:
                edited_sent.append(sort_dict[0][0])
            else:
                edited_sent.append(candidate_word[0])
        else:
            edited_sent.append(candidate_word[0])

    return edited_sent


def read_text(path):
    typo_set = []
    result = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line[0] in ['<', '>']:
            r_line = line[1:].strip().replace('"', '')
            r_line = r_line.replace('·', '')
            typo_set.append(r_line)
        else:
            typo_set.append(line.strip())
            result.append(typo_set)
            typo_set = []
    return result


def edit_test(input_path, output_path, unigram, word_bi_f, word_bi_b, word_tri_f, word_tri_b, syl_tri_f, syl_tri_b,
              min_freq, using_syl_edit=True):
    correct_sent_num = 0
    correct_word_num = 0

    correct_typo_word_num = 0

    no_edit_typo = 0
    error_word_num = 0

    correct_result = ''
    error_result = ''

    sample = read_text(input_path)
    print(len(sample))

    for k, s in enumerate(sample):
        start = time.time()
        typo_pos = s[2].split()
        typo_pos_int = []

        for p in typo_pos:
            typo_pos_int.append(int(p))

        try:
            edit_sent_result = edit_sent(s[1], unigram, min_freq, word_bi_f, word_bi_b, word_tri_f,
                                         word_tri_b, syl_tri_f, syl_tri_b, using_syl_edit)  # Tri추가
            # edit_sent_result = edit_sent(s[1], unigram, min_freq, word_bi_gram, word_bi_gram_r) #어절 바이그램만 사용하는 경우
        except MemoryError:
            continue
        target_sent = s[0].split()
        # 음절 임베딩을 사용한 경우를 찾기 위한 조치
        edited_sent = ' '.join(edit_sent_result)
        if edited_sent == s[0]: #정답 확인
            correct_sent_num += 1
            correct_word_num += len(target_sent)
            correct_typo_word_num += 3
            correct_result += f'<{s[0]}\n>{s[1]}\n{s[2]}\n\n'
        else:
            error_result += f'<{s[0]}\n>{s[1]}\n{s[2]}\n==>{" ".join(edit_sent_result)}\n\n'
            for i, correct_word in enumerate(target_sent):  # 음절 임베딩을 사용한 경우를 찾기 위한 조치
                if i in typo_pos_int:
                    edited_word = edit_sent_result[i]
                    if edited_word == correct_word:
                        correct_typo_word_num += 1
                        correct_word_num += 1
                    else:
                        no_edit_typo += 1
                else:
                    edited_word = edit_sent_result[i]
                    if edited_word == correct_word:
                        correct_word_num += 1
                    else:
                        error_word_num += 1
        print(f'{k} time : {time.time() - start}')
        # if (k+1) % 100 == 0:
        # break
    with open(output_path + 'correct_edit_sent_ts.txt', 'w', encoding='utf-8') as correct_f:
        correct_f.write(correct_result)
    with open(output_path + 'error_edit_sent_ts.txt', 'w', encoding='utf-8') as error_f:
        error_f.write(error_result)

    logging.info(msg='correct sentence num: ' + str(correct_sent_num))
    logging.info(msg='correct word num: ' + str(correct_word_num))
    logging.info(msg='correct edit typo word num: ' +
                 str(correct_typo_word_num))
    logging.info(msg='decorrect edit typo word num: ' + str(no_edit_typo))
    logging.info(
        msg='edit original word(word that no need to edit) num: ' + str(error_word_num))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--typo_text_path", default='./typo_text.txt', type=str, required=True, help="typo text path"
    )
    parser.add_argument(
        "--output_path", default='./', type=str, required=True, help="output result folder path"
    )
    parser.add_argument(
        "--using_syl_edit",
        action="store_true",
        help="using syllable edit module",
    )
    parser.add_argument(
        "--unigram_path", default='./unigram_150.json', type=str, required=True, help="unigram lm path"
    )
    parser.add_argument(
        "--word_bigram_f_path", default='./word_bi_result.json', type=str, required=True,
        help="word bigram forward lm path"
    )
    parser.add_argument(
        "--word_bigram_b_path", default='./word_bi_result_reverse.json', type=str, required=True,
        help="word bigram backward lm path"
    )
    parser.add_argument(
        "--word_trigram_f_path", default='./Wtri_M.json', type=str, required=True, help="word trigram forward lm path"
    )
    parser.add_argument(
        "--word_trigram_b_path", default='./Wtri_M_r.json', type=str, required=True,
        help="word trigram backward lm path"
    )
    parser.add_argument(
        "--syl_trigram_f_path", default='./f_result_tri.json', type=str, required=True,
        help="syllable trigram forward lm path"
    )
    parser.add_argument(
        "--syl_trigram_b_path", default='./b_result_tri.json', type=str, required=True,
        help="syllable trigram backward lm path"
    )
    parser.add_argument(
        "--unigram_min_freq", default=1, type=int, required=True, help="candidate unigram minimum frequency"
    )

    args = parser.parse_args()

    unigram = load_json(args.unigram_path)
    word_bi_f = load_json(args.word_bigram_f_path)
    word_bi_b = load_json(args.word_bigram_b_path)
    word_tri_f = load_json(args.word_trigram_f_path)
    word_tri_b = load_json(args.word_trigram_b_path)

    if args.using_syl_edit:
        syl_tri_f = load_json(args.syl_trigram_f_path)
        syl_tri_b = load_json(args.syl_trigram_b_path)

    edit_test(args.typo_text_path, args.output_path, unigram, word_bi_f, word_bi_b, word_tri_f, word_tri_b, syl_tri_f,
              syl_tri_b, args.unigram_min_freq, args.using_syl_edit)


if __name__ == "__main__":
    main()
