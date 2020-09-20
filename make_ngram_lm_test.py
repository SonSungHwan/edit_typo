import unittest
from collections import Counter
from collections import defaultdict
import make_ngram_lm as mnl
from hangul_char_util import load_json

SAMPLE_TEXT = [
    '언제나 같은 시각에 오는게 더 좋을 거야.',
    '이를테면, 네가 오후 4시에 온다면, 난 3시부터 행복해지기 시작할거야.',
    '시간이 갈수록 난 점점 더 행복해지겠지.'
]


class Make_Ngram_lm_Test(unittest.TestCase):


    def test_get_sent_list(self):
        c = mnl.get_sent_list(SAMPLE_TEXT[1], 2)
        self.assertEqual(
            c,
            ['<<SOS>>', '이를테면,', '네가', '오후', '<<NUM>>시에', '온다면,', '난',
             '<<NUM>>시부터', '행복해지기', '시작할거야.', '<<EOS>>']
        )


    def test_get_word_list(self):
        c = mnl.get_word_list('\n'.join(SAMPLE_TEXT))
        self.assertEqual(
            c,
            ['언제나', '같은', '시각에', '오는게', '더', '좋을', '거야.', '이를테면,',
             '네가', '오후', '4시에', '온다면,', '난', '3시부터', '행복해지기',
             '시작할거야.', '시간이', '갈수록', '난', '점점', '더', '행복해지겠지.']
        )


    def test_syllable_add_SEtoken(self):
        c = mnl.syllable_add_SEtoken('3시부터', 3)
        self.assertEqual(
            c,
            ['<<SOS>>', '<<SOS>>', '<<NUM>>', '시', '부', '터', '<<EOS>>', '<<EOS>>']
        )


    def test_n_gram(self):
        c0 = mnl.n_gram(mnl.get_sent_list(SAMPLE_TEXT[1], 2), 2)
        c1 = mnl.n_gram(mnl.syllable_add_SEtoken('3시부터', 3), 3)
        self.assertEqual(
            c0,
            [('<<SOS>>', '이를테면,'), ('이를테면,', '네가'), ('네가', '오후'), ('오후', '<<NUM>>시에'),
             ('<<NUM>>시에', '온다면,'), ('온다면,', '난'), ('난', '<<NUM>>시부터'),
             ('<<NUM>>시부터', '행복해지기'), ('행복해지기', '시작할거야.'), ('시작할거야.', '<<EOS>>')]
        )
        self.assertEqual(
            c1,
            [('<<SOS>>', '<<SOS>>', '<<NUM>>'), ('<<SOS>>', '<<NUM>>', '시'), ('<<NUM>>', '시', '부'),
             ('시', '부', '터'), ('부', '터', '<<EOS>>'), ('터', '<<EOS>>', '<<EOS>>')]
        )


    def test_fwbw_n_gram(self):
        c0, c1 = mnl.fwbw_n_gram(mnl.n_gram(
            mnl.get_sent_list(SAMPLE_TEXT[1].strip(), 2), 2), syl_ngram=False)
        c2, c3 = mnl.fwbw_n_gram(mnl.n_gram(
            mnl.syllable_add_SEtoken('3시부터', 3), 3), syl_ngram=True)
        self.assertEqual(
            c0,
            [('<<SOS>>', '이를테면,'), ('이를테면,', '네가'), ('네가', '오후'), ('오후', '<<NUM>>시에'),
             ('<<NUM>>시에', '온다면,'), ('온다면,', '난'), ('난', '<<NUM>>시부터'),
             ('<<NUM>>시부터', '행복해지기'), ('행복해지기', '시작할거야.')]
        )
        self.assertEqual(
            c1,
            [('이를테면,', '네가'), ('네가', '오후'), ('오후', '<<NUM>>시에'), ('<<NUM>>시에', '온다면,'),
             ('온다면,', '난'), ('난', '<<NUM>>시부터'), ('<<NUM>>시부터', '행복해지기'),
             ('행복해지기', '시작할거야.'), ('시작할거야.', '<<EOS>>')]
        )
        self.assertEqual(
            c2,
            [('<<SOS>>', '<<NUM>>', '시'), ('<<NUM>>', '시', '부'), ('시', '부', '터')]
        )
        self.assertEqual(
            c3,
            [('시', '부', '터'), ('부', '터', '<<EOS>>'), ('터', '<<EOS>>', '<<EOS>>')]
        )


    def test_preprocessing(self):
        c0, c1 = mnl.preprocessing(SAMPLE_TEXT, 2, syl_ngram=False)
        c2, c3 = mnl.preprocessing(SAMPLE_TEXT, 2, syl_ngram=True)
        self.assertEqual(
            c0,
            [[('<<SOS>>', '언제나'), ('언제나', '같은'), ('같은', '시각에'), ('시각에', '오는게'),
              ('오는게', '더'), ('더', '좋을'), ('좋을', '거야.')],
             [('<<SOS>>', '이를테면,'), ('이를테면,', '네가'), ('네가', '오후'), ('오후', '<<NUM>>시에'),
              ('<<NUM>>시에', '온다면,'), ('온다면,', '난'), ('난', '<<NUM>>시부터'),
              ('<<NUM>>시부터', '행복해지기'), ('행복해지기', '시작할거야.')],
             [('<<SOS>>', '시간이'), ('시간이', '갈수록'), ('갈수록', '난'),
              ('난', '점점'), ('점점', '더'), ('더', '행복해지겠지.')]]
        )
        self.assertEqual(
            c1,
            [[('언제나', '같은'), ('같은', '시각에'), ('시각에', '오는게'), ('오는게', '더'), ('더', '좋을'),
              ('좋을', '거야.'), ('거야.', '<<EOS>>')],
             [('이를테면,', '네가'), ('네가', '오후'), ('오후', '<<NUM>>시에'), ('<<NUM>>시에', '온다면,'),
              ('온다면,', '난'), ('난', '<<NUM>>시부터'), ('<<NUM>>시부터', '행복해지기'),
              ('행복해지기', '시작할거야.'), ('시작할거야.', '<<EOS>>')],
             [('시간이', '갈수록'), ('갈수록', '난'), ('난', '점점'), ('점점', '더'),
              ('더', '행복해지겠지.'), ('행복해지겠지.', '<<EOS>>')]]
        )
        self.assertEqual(
            c2,
            [[('<<SOS>>', '언'), ('언', '제'), ('제', '나')],
             [('<<SOS>>', '같'), ('같', '은')],
             [('<<SOS>>', '시'), ('시', '각'), ('각', '에')],
             [('<<SOS>>', '오'), ('오', '는'), ('는', '게')],
             [('<<SOS>>', '더')],
             [('<<SOS>>', '좋'), ('좋', '을')],
             [('<<SOS>>', '거'), ('거', '야')],
             [('<<SOS>>', '이'), ('이', '를'), ('를', '테'), ('테', '면')],
             [('<<SOS>>', '네'), ('네', '가')],
             [('<<SOS>>', '오'), ('오', '후')],
             [('<<NUM>>', '시'), ('시', '에')],
             [('<<SOS>>', '온'), ('온', '다'), ('다', '면')],
             [('<<SOS>>', '난')],
             [('<<NUM>>', '시'), ('시', '부'), ('부', '터')],
             [('<<SOS>>', '행'), ('행', '복'), ('복', '해'), ('해', '지'), ('지', '기')],
             [('<<SOS>>', '시'), ('시', '작'), ('작', '할'), ('할', '거'), ('거', '야')],
             [('<<SOS>>', '시'), ('시', '간'), ('간', '이')],
             [('<<SOS>>', '갈'), ('갈', '수'), ('수', '록')],
             [('<<SOS>>', '난')],
             [('<<SOS>>', '점'), ('점', '점')],
             [('<<SOS>>', '더')],
             [('<<SOS>>', '행'), ('행', '복'), ('복', '해'), ('해', '지'), ('지', '겠'), ('겠', '지')]]
        )
        self.assertEqual(
            c3,
            [[('언', '제'), ('제', '나'), ('나', '<<EOS>>')],
             [('같', '은'), ('은', '<<EOS>>')],
             [('시', '각'), ('각', '에'), ('에', '<<EOS>>')],
             [('오', '는'), ('는', '게'), ('게', '<<EOS>>')],
             [('더', '<<EOS>>')],
             [('좋', '을'), ('을', '<<EOS>>')],
             [('거', '야'), ('야', '.')],
             [('이', '를'), ('를', '테'), ('테', '면'), ('면', ',')],
             [('네', '가'), ('가', '<<EOS>>')],
             [('오', '후'), ('후', '<<EOS>>')],
             [('시', '에'), ('에', '<<EOS>>')],
             [('온', '다'), ('다', '면'), ('면', ',')],
             [('난', '<<EOS>>')],
             [('시', '부'), ('부', '터'), ('터', '<<EOS>>')],
             [('행', '복'), ('복', '해'), ('해', '지'), ('지', '기'), ('기', '<<EOS>>')],
             [('시', '작'), ('작', '할'), ('할', '거'), ('거', '야'), ('야', '.')],
             [('시', '간'), ('간', '이'), ('이', '<<EOS>>')],
             [('갈', '수'), ('수', '록'), ('록', '<<EOS>>')], [('난', '<<EOS>>')],
             [('점', '점'), ('점', '<<EOS>>')],
             [('더', '<<EOS>>')],
             [('행', '복'), ('복', '해'), ('해', '지'), ('지', '겠'), ('겠', '지'), ('지', '.')]]
        )


    def test_calc_n_gram(self):
        c0, c1 = mnl.calc_n_gram(SAMPLE_TEXT, 2, defaultdict(Counter),
                                 defaultdict(Counter), syl_ngram=False, last=True)
        c2, c3 = mnl.calc_n_gram(SAMPLE_TEXT, 2, defaultdict(Counter),
                                 defaultdict(Counter), syl_ngram=True, last=True)
        c0_expected = load_json('./test_data/mnl_unittest_forward_2ngram_word_dic.json')
        c1_expected = load_json('./test_data/mnl_unittest_backward_2ngram_word_dic.json')
        c2_expected = load_json('./test_data/mnl_unittest_forward_2ngram_syl_dic.json')
        c3_expected = load_json('./test_data/mnl_unittest_backward_2ngram_syl_dic.json')

        for k in c0_expected.keys():
            for kk, vv in c0_expected[k].items():
                self.assertEqual(round(vv, 5), round(c0[k][kk], 5))
        for k in c1_expected.keys():
            for kk, vv in c1_expected[k].items():
                self.assertEqual(round(vv, 5), round(c1[k][kk], 5))
        for k in c2_expected.keys():
            for kk, vv in c2_expected[k].items():
                self.assertEqual(round(vv, 5), round(c2[k][kk], 5))
        for k in c3_expected.keys():
            for kk, vv in c3_expected[k].items():
                self.assertEqual(round(vv, 5), round(c3[k][kk], 5))

if __name__ == '__main__':
    unittest.main()