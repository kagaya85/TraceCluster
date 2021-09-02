import re
from typing import List


def boolStr2Int(str: str) -> int:
    if str == 'True' or str == 'true' or str == 'TRUE':
        return 1
    else:
        return 0


def boolStr2Bool(str: str) -> bool:
    if str == 'True' or str == 'true' or str == 'TRUE':
        return True
    else:
        return False


def hump2snake(s: str) -> str:
    '''
    驼峰形式字符串转成下划线形式
    '''
    # 匹配正则，匹配小写字母和大写字母的分界位置
    p = re.compile(r'([a-z]|\d)([A-Z])')
    sub = re.sub(p, r'\1_\2', s).lower()
    return sub


def wordSplit(s: str, words: List[str]) -> List[str]:
    if len(s) == 0 or len(words) == 0:
        return []

    ans = [s]
    for w in words:
        tmp = ans
        ans = []
        for substr in tmp:
            idx = substr.find(w)

            while idx != -1:
                if idx > 0:
                    ans.append(substr[:idx])
                ans.append(w)
                substr = substr[idx+len(w):]
                idx = substr.find(w)

            if len(substr) > 0:
                ans.append(substr)

    return ans
