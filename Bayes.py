# encoding = utf-8
# Author:supersunshinefk
# 实现功能：运用朴素贝叶斯算法进行邮件过滤
# Time:2018/6/7 15：28
import numpy as np


def train_set(emails):
    # ###################################################清洗数据###############################################
    # 第一步 1.对emails 样本集进行去重整合成train列表
    # ！！！！需要注意的是：set集合内部元素的排序是无序的，所以后面程序zero_one列表（第60行的zero_one输出）
    # 的值不一样（是动态的），因为每一次运行以后都有顺序都有变！！！！！！
    train = set()
    for email in emails:
        for element in email:
            train.add(element)

    # 第一步 2.过滤掉间段的单词,包括语气助词单词 word列表
    # set.discard()
    # discard(ele)将一个元素从集合中移除。如果元素不存在于集合中，它不会抛出KeyError；如果存在于集合中，则会移除数据并返回None
    words = ['is', 'so', 'to', 'i', 'I', 'am', 'are', 'in', 'on', 'how', 'my', 'him']
    for word in words:
        train.discard(word)
    # set集合转换成lis类型
    train = list(train)

    # 定义一个空的索引矩阵容器 用于存储emails中每个元素在train中出现的索引位置
    index_list = list()

    # 第二步 定义个大容器分存放上表中每一份邮件的分类值（0，1）
    zero_one = list()
    for k in range(len(emails)):
        zero_one.append(np.zeros(len(train)).tolist())

        index_list.append([])


    for email_ in emails:
        j = emails.index(email_)    # 精髓！
        for i in email_:
            if i in train:
                index_i = train.index(i)
                index_list[j].append(index_i)

    # 转换成词向量  根据index索引列表将zero_one列表元素置为 1 表示在train中出现过的
    for zero_one_line1, index_line in zip(zero_one, index_list):
        for index_element in index_line:
            zero_one_line1[index_element] = 1

    return zero_one, train


def algorithm(zero_one, labels):

    # ####################################算法实现#########################################################
    # 第一步：计算P(c) 垃圾邮件和非垃圾邮件的概率 P_c_0, P_c_1
    all_labels = len(labels)
    count_1 = 0
    count_0 = 0
    for label in labels:
        if label == 0:
            count_0 = count_0 + 1
        if label == 1:
            count_1 += 1

    p_c_0 = count_0/all_labels   # 非垃圾邮件的概率
    p_c_1 = count_1/all_labels   # 垃圾邮件的概率

    # 第二步  计算P(w|c)的概率   P（w1|c1） P(w2|c2)
    # 把垃圾邮件归为一类，成一个列表矩阵
    rubbish_list = list()
    # 把非垃圾邮件归为一类，成一个列表矩阵
    no_rubbish_list = list()
    for zero_one_line2, labels2 in zip(zero_one, labels):
        if labels2 == 0:
            no_rubbish_list.append(zero_one_line2)
        if labels2 == 1:
            rubbish_list.append(zero_one_line2)

    # 邮件进行归类以后进行合并， 如垃圾邮件的每一行的列进行相加  合并
    rubbish_list_add = np.sum(rubbish_list, axis=0)
    no_rubbish_list_add = np.sum(no_rubbish_list, axis=0)

    # 对 行的每一个元素进行相加 求出总和  为一个值
    rubbish_list_all_add = np.sum(rubbish_list_add)
    no_rubbish_list_all_add = np.sum(no_rubbish_list_add)

    # 求出每个元素的出现次数除以总和  求出其概率
    # P(w1|c1)
    rubbish_element_prob = rubbish_list_add/rubbish_list_all_add
    # P(w2|c2)
    no_rubbish_element_prob = no_rubbish_list_add/no_rubbish_list_all_add

    return p_c_0, p_c_1, rubbish_element_prob, no_rubbish_element_prob


def test_set(test_emails, train, p_c_0, p_c_1, rubbish_element_prob, no_rubbish_element_prob):
    test_w = np.zeros(len(train))
    index_j_list = list()
    for test_email in test_emails:
        if test_email in train:
            index_j = train.index(test_email)
            index_j_list.append(index_j)

    for index_j_list_each in index_j_list:
        test_w[index_j_list_each] = 1

    test_w_c_rubbish = np.multiply(test_w, rubbish_element_prob)  # P_c_1
    test_w_c_no_rubbish = np.multiply(test_w, no_rubbish_element_prob)  # P_c_0

    test_rubbish = np.round(test_w_c_rubbish.tolist(), 3)
    test_no_rubbish = np.round(test_w_c_no_rubbish.tolist(), 3)

    # 原式 P(c1|wi) = [P(w1|c1)*P(w2|c1)*P(w3|c1)*P(w4|c1)...]*P(c1)  此处的分母P(w)省略
    test_sum_rubbish = 0
    for test_rubbish_i in test_rubbish:
        if test_rubbish_i > 0:
             test_sum_rubbish += np.log(test_rubbish_i)
    test_sum_rubbish += np.log(p_c_1)

    # 原式 P(c0|wi) = [P(w1|c0)*P(w2|c0)*P(w3|c0)*P(w4|c0)...]*P(c0)  此处的分母P(w)省略  取对数log10（）
    # log10(P(c0|wi)) = [log10(P(w1|c0)) + log10(P(w2|c0)) + log10(P(w3|c0)) + log10(P(w4|c0))...] + log10(P(c0))
    # 因为是比较垃圾邮件和非垃圾邮件概率大小 ，而其分母P(w)是相同的可以同时忽略！
    test_sum_no_rubbish = 0
    for test_no_rubbish_i in test_no_rubbish:
        if test_no_rubbish_i > 0:
            test_sum_no_rubbish += np.log(test_no_rubbish_i)
    test_sum_no_rubbish += np.log(p_c_0)

    # print('test_sum_rubbish>>>', test_sum_rubbish)
    # print('test_sum_no_rubbish>>>', test_sum_no_rubbish)

    name = input('您好！请输入您的姓名：')
    if test_sum_no_rubbish < test_sum_rubbish:
        print('尊敬的{0}先生，该邮件为非垃圾邮件！'.format(name))
    if test_sum_no_rubbish > test_sum_rubbish:
        print('尊敬的{0}先生，该邮件为垃圾邮件！'.format(name))


if __name__ == '__main__':
    # 样本集
    emails = [
        ['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'ny', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    # 对emails列表中的每句话进行0,1分类
    # 0 表示非垃圾邮件  1 表示垃圾邮件
    labels = [0, 1, 0, 1, 0, 1]   

    zero_one, train = train_set(emails)

    p_c_0, p_c_1, rubbish_element_prob, no_rubbish_element_prob = algorithm(zero_one, labels)

    # 对测试邮件 进行分类
    test_emails = ['love', 'garbage', 'stupid']
    # test_emails = ['garbage', 'help', 'dog', 'stupid', 'food']

    test_set(test_emails, train, p_c_0, p_c_1, rubbish_element_prob, no_rubbish_element_prob)
