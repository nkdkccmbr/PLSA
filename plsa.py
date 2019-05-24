# -*- encoding:utf-8 -*- #
import math
import numpy as np

# 同時確率 P(D,W) -> p_dw
# 事後確率 P(D|Z) -> pd_z

THRESHOLD = 1.0
PRINT_NUM = 10
NLL_MAX   = 0

BETA_MOMENTUM = 0.99

STOP_WORDS = ['、', '。', '\n', '(', ')', '（', '）', '「', '」', '[', ']', '，', '．',
              '「', '」', '・', '', '', '', '', '', '', '', 
              'て', 'に', 'を', 'は', 'と', 'も', 'ない', 'よう', 'に', 'さ',
              'し', 'が', 'など', 'れ', 'た', 'が', 'や', 'せ', 'または', 'また',
              'で', 'ある', 'この', 'その', 'あの', 'どの', 'こと', 'な', 'の',
              'ら', 'い', 'ん', 'られる', 'ず', 'という', 'なり', 'とき', 'により', 'いう',
              'する', 'した', 'れる', 'において', 'そして', 'ただ', 'み', 'もの', 'まし', 'たり',
              'なお', 'ほか', 'いる', '', '', '', '', '', '', ]

class PLSA():
    # 単語と文書の行列(np.array)とトピック数
    def __init__(self, MATRIX, K):
        # トピックZの数
        self.K = K
        # 文書Dの数
        self.N = MATRIX.shape[0]
        # 単語Wの数
        self.M = MATRIX.shape[1]
    
        # 頻度
        self.n_dw = MATRIX
    
        # 確率
        self.pw_z  = np.random.rand(self.M, self.K)
        self.pd_z  = np.random.rand(self.N, self.K)
        self.pz    = np.random.rand(self.K)
        self.pz_dw = np.random.rand(self.K, self.N, self.M)

        # TEMのパラメータ
        self.beta = 1.0

    # 対数尤度の下限
    def log_likelihood_minimum(self):
        l = 0
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.K):
                    l += self.n_dw[j, i] * self.pz_dw[k, j, i] * \
                      math.log(self.pw_z[i, k] * self.pd_z[j, k] * self.pz[k] + 1e-5)
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.K):
                    l -= self.n_dw[j, i] * self.pz_dw[k, j, i] * math.log(self.pz_dw[k, j, i] + 1e-5)
        return l

    def e_step(self):
        for i in range(self.M):
            for j in range(self.N):
                sum_k = 0
                for k in range(self.K):
                    sum_k += self.pz[k] * (self.pw_z[i, k] * self.pd_z[j, k])**self.beta
                for k in range(self.K):
                    self.calc_pz_dw(i, j, k, sum_k)

    def calc_pz_dw(self, i, j, k, sum_k):
        self.pz_dw[k, j, i] = (self.pz[k] * (self.pw_z[i, k] * self.pd_z[j, k])**self.beta) / sum_k
        
    def m_step(self):
        for i in range(self.M):
            for k in range(self.K):
                self.calc_pw_z(i, k)
        for j in range(self.N):
            for k in range(self.K):
                self.calc_pd_z(j, k)
        for k in range(self.K):
            self.calc_z(k)

    def calc_pw_z(self, i, k):
        sum_n = 0
        for j in range(self.N):
            sum_n += self.n_dw[j, i] * self.pz_dw[k, j, i]
        sum_mn = 0
        for i_ in range(self.M):
            for j in range(self.N):
                sum_mn += self.n_dw[j, i_] * self.pz_dw[k, j, i_]
        self.pw_z[i, k] = sum_n / sum_mn
    
    def calc_pd_z(self, j, k):
        sum_m = 0
        for i in range(self.M):
            sum_m += self.n_dw[j, i] * self.pz_dw[k, j, i]
        sum_mn = 0
        for i in range(self.M):
            for j_ in range(self.N):
                sum_mn += self.n_dw[j_, i] * self.pz_dw[k, j_, i]
        self.pd_z[j, k] = sum_m / sum_mn
    
    def calc_z(self, k):
        sum_mn1 = 0
        for i in range(self.M):
            for j in range(self.N):
                sum_mn1 += self.n_dw[j, i] * self.pz_dw[k, j, i]
        sum_mn2 = 0
        for i in range(self.M):
            for j in range(self.N):
                sum_mn2 += self.n_dw[j, i]
        self.pd_z[j, k] = sum_mn1 / sum_mn2
        
    def em_algorithm(self):
        pre_log_likelihood = NLL_MAX
        while True:
            self.e_step()
            self.m_step()
            log_likelihood = self.log_likelihood_minimum()
            print(log_likelihood)
            if abs(log_likelihood - pre_log_likelihood) < THRESHOLD:
                break
            pre_log_likelihood = log_likelihood
            self.beta *= BETA_MOMENTUM

if __name__ == '__main__':
    TOPIC_NUM = 3
    # テキストファイルに切り出す
    # 1行1文
    doc = []
    with open('document.txt', 'r') as f:
        for line in f:
            doc.append(line)

    # 分かち書き
    from janome.tokenizer import Tokenizer
    t = Tokenizer()
    doc_tokens = [t.tokenize(d) for d in doc]
    sentences = [[token.surface for token in sentence_tokens if token.surface not in STOP_WORDS] for sentence_tokens in doc_tokens]

    # BOWを作る
    bow = list({word for sentence in sentences for word in sentence})
    # sentence_word_id_list = [[bow.index(word) for word in word_list] for word_list in sentence_word_list]
    # print(bow)

    # print(sentence_word_id_list)
    # word_list = [bow[w] for sw in sentence_word_id_list for w in sw]

    # 文書をBOWの頻度行列に直す
    MATRIX = np.array([[sentence.count(word) for word in bow] for sentence in sentences])

    # pLSAの構築
    plsa = PLSA(MATRIX, TOPIC_NUM)

    # 学習
    plsa.em_algorithm()

    # 単語ごとのトピックに属する確率
    print('単語-トピック')
    for i in range(plsa.pw_z.shape[1]):
        print('Topic:', i)
        dic = {}
        for j in range(plsa.pw_z.shape[0]):
            dic[j] = (bow[j], plsa.pw_z[j][i])
        sorted_list = sorted(dic.items(), key=lambda x:x[1][1], reverse=True)
        for j, s in enumerate(sorted_list):
            if j > PRINT_NUM:
                break
            print(s[1])

    # 文書ごとのトピックに属する確率
    print('文書-トピック')
    print(plsa.pd_z)
    for i in range(plsa.pd_z.shape[1]):
        print('Topic:', i)
        dic = {}
        for j in range(plsa.pd_z.shape[0]):
            dic[j] = (doc[j], plsa.pd_z[j][i])
        sorted_list = sorted(dic.items(), key=lambda x:x[1][1], reverse=True)
        for j, s in enumerate(sorted_list):
            if j > PRINT_NUM:
                break
            print(s[1])
