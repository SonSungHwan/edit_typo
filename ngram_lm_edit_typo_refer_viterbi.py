# edit_typo code lib

import re
from operator import itemgetter
import copy
import time
import argparse
import logging
from itertools import product

from hangul_char_util import load_json, split_syllable_char, join_jamos, check_syllable, check_han_word
from generate_typo import change_c
from ngram_lm_edit_typo import syllable_edit_word, make_sent_combi_list

NUMBER = '<<NUM>>'
ENG = '<<ENG>>'
SENT_START = '<<SOS>>'
SENT_END = '<<EOS>>'

TOTAL_CHAR = set(['ㄲ', 'ㅆ', 'ㄱ', 'ㅅ', 'ㅛ', 'ㅕ', 'ㄹ', 'ㅎ', 'ㅗ', 'ㅊ', 'ㅍ', 'ㅠ', 'ㄺ', 'ㄽ', 'ㅃ', 'ㅉ',
                  'ㄸ', 'ㅒ', 'ㅖ', 'ㅂ', 'ㅈ', 'ㄷ', 'ㅁ', 'ㄴ', 'ㅇ', 'ㅋ', 'ㅌ', 'ㅑ', 'ㅐ', 'ㅔ', 'ㅓ', 'ㅏ',
                  'ㅣ', 'ㅜ', 'ㅡ', 'ㅘ', 'ㅚ', 'ㅟ', 'ㅢ'])


def replace_num(word):
    if re.search(r'\d+', word):
        word = re.sub(r'\d+', NUMBER, word)
    return word


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


def context_window_list(wcl, word_bi_f, word_bi_b):
    score_list = []
    word_candis_score_buff = {}

    for i, candidate in enumerate(wcl[1:-1]):
        if candidate[0][1] == 1.0 or i == 0:
            if i == 0:  # 어절 바이그램은 첫 어절도 양변을 보도록 수정(19-09-10)
                check = 0.
                for cw in candidate:  # 전부 0이 아니라면 바이그램 수치 사용
                    for aft_word in wcl[i + 2]:
                        try:
                            if replace_num(cw[0]) in word_bi_b[replace_num(aft_word[0])]:
                                if cw[0] not in word_candis_score_buff:
                                    word_candis_score_buff[cw[0]] = {}
                                    word_candis_score_buff[cw[0]][SENT_START] = 0.
                                try:
                                    word_candis_score_buff[cw[0]][SENT_START] += word_bi_b[replace_num(aft_word[0])][
                                        replace_num(cw[0])]
                                except KeyError:
                                    pass
                        except KeyError:
                            pass

                    if cw[0] in word_candis_score_buff:
                        word_candis_score_buff[cw[0]][SENT_START] /= len(word_candis_score_buff)
                        check += word_candis_score_buff[cw[0]][SENT_START]
                if check == 0.:
                    word_candis_score_buff = {}
                    max_index = max(candidate, key=lambda x: x[1])
                    candi_word = max_index[0]
                    word_candis_score_buff[candi_word] = {}
                    word_candis_score_buff[candi_word][SENT_START] = 1.
                score_list.append(word_candis_score_buff)
                word_candis_score_buff = {}

            else:
                max_index = max(candidate, key=lambda x: x[1])
                candi_word = max_index[0]
                word_candis_score_buff[candi_word] = {}

                for pre_word in wcl[i]:
                    word_candis_score_buff[candi_word][pre_word[0]] = 1.

                score_list.append(word_candis_score_buff)
                word_candis_score_buff = {}
        else:
            candi_score_temp = []
            check = True
            for word in candidate:
                word_candis_score_buff[word[0]] = {}
                for pre_word in wcl[i]:
                    b_p = 0.
                    try:
                        f_p = word_bi_f[replace_num(
                            pre_word[0])][replace_num(word[0])]
                    except KeyError:
                        f_p = 0.

                    for aft_word in wcl[i + 2]:
                        try:
                            b_p += word_bi_b[replace_num(aft_word[0])
                                             ][replace_num(word[0])]
                        except KeyError:
                            b_p += 0.

                    candi_score_temp.append(
                        (word[0], pre_word[0], (f_p + (b_p / len(wcl[i + 2]))) / 2))
                    word_candis_score_buff[word[0]][pre_word[0]] = (
                        f_p + (b_p / len(wcl[i + 2]))) / 2
                candi_score_temp = sorted(candi_score_temp, key=lambda x: x[2], reverse=True)
                if candi_score_temp[0][2] > 0.:
                    check = False
                candi_score_temp = []
            # print(score_list)
            if check:  # 바이그렘 정보가 없을 경우 빈도수 최대?, 첫 임력?
                candi_word_list = word_candis_score_buff.keys()
                for cw in candi_word_list:
                    for key in word_candis_score_buff[cw].keys():
                        word_candis_score_buff[cw][key] = 1.
                score_list.append(word_candis_score_buff)
                word_candis_score_buff = {}
            else:
                for kk in word_candis_score_buff.keys(): #Check
                    sorted_buff = sorted(
                        word_candis_score_buff[kk].items(), key=itemgetter(1), reverse=True)
                    if sorted_buff[0][1] == 0.:
                        del word_candis_score_buff[kk]

                score_list.append(word_candis_score_buff)
                word_candis_score_buff = {}

    return score_list


def viterbi_edit(sent, unigram, word_bi_f, word_bi_b, syl_tri_f, syl_tri_b, min_freq, using_syl_edit):
    word_combi_list = make_sent_combi_list(
        sent, unigram, syl_tri_f, syl_tri_b, min_freq, using_syl_edit)
    # rute = bigram_score_list_dict(word_combi_list, word_bigram) #단방향
    rute = context_window_list(word_combi_list, word_bi_f, word_bi_b)  # 양방향
    paths = [] #경로들을 추려서 저장
    original_tokens = sent.strip().split()
    # print(rute)
    for i, node_list in enumerate(word_combi_list[1:-1]):
        # path = [[], 1.]
        all_path_buff = []
        for node in node_list:
            # candi_keys는 실제 타겟 단어로
            # print(node)
            try:
                pre_keys = list(rute[i][node[0]].keys())
            except KeyError:
                continue

            for j, p in enumerate(paths):
                if len(p[0]) < i:
                    del paths[j]

            if len(paths) > 0:  # 0이 아닌 path를 추가 시키도록 수정해야함
                for k, p in enumerate(paths):
                    temp_path = []
                    c_paths = copy.deepcopy(paths)
                    last_word = c_paths[k][0][-1]
                    for pk in pre_keys:
                        # print(last_word, pk, node[0])
                        if last_word == pk and rute[i][node[0]][pk] > 0.:
                            temp_p = p[0][:]
                            temp_p.append(node[0])
                            temp_path.append(
                                [temp_p, p[1] * rute[i][node[0]][pk]])
                        # else:
                        # print(pk, node[0])
                        # print(rute[i][node[0]][pk])
                        # print('-----')
                    # if len(temp_path) > 0:
                    if temp_path not in all_path_buff:
                        all_path_buff.extend(temp_path)

                # if len(all_path_buff) == 0:
                # for k, p in enumerate(paths):
                # 'dd'
            else:
                for pk in pre_keys:
                    if rute[i][node[0]][pk] > 0.:
                        paths.append([[node[0]], 1. * rute[i][node[0]][pk]])

            if len(all_path_buff) == 0:
                for k, p in enumerate(paths):
                    temp_p = []
                    c_paths = copy.deepcopy(paths)
                    last_word = c_paths[k][0][-1]
                    for pk in pre_keys:
                        if last_word == pk:
                            temp_p = p[0][:]
                            temp_p.append(node[0])
                            paths.append([temp_p, p[1]])
            else:
                for ttp in all_path_buff:
                    if len(ttp[0]) == i + 1 and ttp not in paths:
                        paths.append(ttp)

        temp_paths = []  # 업데이트 되지 않은 이전 path의 삭제
        selected_target_list = []
        for rute_path in paths:
            if len(rute_path[0]) == (i + 1):
                temp_paths.append(rute_path)
        paths = temp_paths

        # 업데이트 된 path 중 같은 target을 선택한 최대 값의 path만 추리도록 조정(수정 요망 19-09-10)
        temp_paths = []
        # print(paths, '==>> 추리기 전 path')
        for rute_path in paths:
            if rute_path[0][-1] not in selected_target_list:
                selected_target_list.append(rute_path[0][-1])
                temp_paths.append(rute_path)
            else:
                path_index = selected_target_list.index(rute_path[0][-1])
                if temp_paths[path_index][1] < rute_path[1]:
                    temp_paths[path_index] = rute_path
                # '''
                elif temp_paths[path_index][1] == rute_path[1]:  # 값이 동일한 경우 selected_target 이전 word의 빈도수로 결정!
                    if rute_path[0][-2] == original_tokens[i - 1]:
                        temp_paths[path_index] = rute_path

                    elif (unigram[replace_num(temp_paths[path_index][0][-2])] < unigram[replace_num(rute_path[0][-2])]
                          and temp_paths[path_index][0][-2] != original_tokens[i - 1]):
                        temp_paths[path_index] = rute_path
                # '''
        paths = temp_paths

    result = []
    for path in paths:
        if len(sent.split()) == len(path[0]):
            result.append(path)
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result[0]


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


def edit_test(input_path, output_path, unigram, word_bi_f, word_bi_b, syl_tri_f, syl_tri_b, min_freq, using_syl_edit):
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
            edit_sent_result = viterbi_edit(s[1], unigram, word_bi_f, word_bi_b, syl_tri_f, syl_tri_b, min_freq,
                                            using_syl_edit)
        except MemoryError:
            continue
        target_sent = s[0].split()
        # 음절 임베딩을 사용한 경우를 찾기 위한 조치
        if edit_sent_result[0] == target_sent:
            correct_sent_num += 1
            correct_word_num += len(target_sent)
            correct_typo_word_num += 3
            correct_result += f'<{s[0]}\n>{s[1]}\n{s[2]}\n\n'
        else:
            error_result += f'<{s[0]}\n>{s[1]}\n{s[2]}\n==>{" ".join(edit_sent_result[0])}\n\n'
            for i, word in enumerate(target_sent):  # 음절 임베딩을 사용한 경우를 찾기 위한 조치
                if i in typo_pos_int:
                    if edit_sent_result[0][i] == word:
                        correct_typo_word_num += 1
                        correct_word_num += 1
                    else:
                        no_edit_typo += 1
                else:
                    if edit_sent_result[0][i] == word:
                        correct_word_num += 1
                    else:
                        error_word_num += 1
        print(f'{k} time : {time.time() - start}')
        if (time.time() - start) > 10:
            print(s[0])
        # if (k+1) % 100 == 0:
        # break
    with open(output_path + 'correct_edit_sent_vit.txt', 'w', encoding='utf-8') as correct_f:
        correct_f.write(correct_result)
    with open(output_path + 'error_edit_sent_vit.txt', 'w', encoding='utf-8') as error_f:
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

    if args.using_syl_edit:
        syl_tri_f = load_json(args.syl_trigram_f_path)
        syl_tri_b = load_json(args.syl_trigram_b_path)

    edit_test(args.typo_text_path, args.output_path, unigram, word_bi_f, word_bi_b, syl_tri_f, syl_tri_b,
              args.unigram_min_freq, args.using_syl_edit)


if __name__ == "__main__":
    main()
