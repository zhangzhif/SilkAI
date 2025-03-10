'''
Jaccard 距离计算
两个文本相同字符数量/两个文本字符的并集数据
'''
def jacccard(str1,str2):
    return len(set(str1)&set(str2))/len(set(str1)|set(str2))


if __name__ == '__main__':
    str1 = "abcde"
    str2 = "abcdf"
    print(jacccard(str1,str2))


