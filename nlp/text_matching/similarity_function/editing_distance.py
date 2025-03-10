import numpy as np
'''
编辑距离（Edit Distance），也称为莱文斯坦距离（Levenshtein Distance）
是指两个字符串之间，由一个字符串转换成另一个字符串所需的最少编辑操作次数。
允许的编辑操作包括插入一个字符、删除一个字符和替换一个字符。

动态规划shi
'''
def editing_distance(str1,str2):
    matrix = np.zeros((len(str1) + 1, len(str2) + 1))
    for i in range(len(str2) + 1):
        matrix[i][0] = i
    for j in range(len(str2) + 1):
        matrix[0][j] = j
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    edit_distance = matrix[len(str1)][len(str2)]
    return 1 - edit_distance / max(len(str1), len(str2))
