import re
import os
from typing import List

from torch_geometric import data

embedding_filename = "embedding.json"


def boolStr2Int(str: str) -> int:
    if str in ['FALSE', 'False', 'false', '0', 0]:
        return 0
    else:
        return 1


def any2bool(x) -> bool:
    if x in ['FALSE', 'False', 'false', '0', 0]:
        return False
    else:
        return True


def int2Bool(n: int) -> bool:
    if n == 0:
        return False
    else:
        return True


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

            if len(substr) > 0 and substr != 's':
                ans.append(substr)

    return ans


def mergeDict(x: dict, y: dict):
    return {**x, **y}


def getNewFile(dirpath: str) -> List[str]:
    list = os.listdir(dirpath)
    filelist = []
    for x in list:
        if not os.path.isdir(x):
            filelist.append(x)

    filelist.sort(key=lambda filename: os.path.getmtime(
        os.path.join(dirpath, filename)))

    if len(filelist) > 0:
        return os.path.join(dirpath, filelist[-1])

    return ""


def getDatafiles(dirpath: str) -> List[str]:
    list = os.listdir(dirpath)
    datafiles = []
    for name in list:
        if name == embedding_filename:
            continue

        filename = os.path.join(dirpath, name)
        if not os.path.isdir(filename):
            datafiles.append(filename)

    return datafiles


def getNewDir(datapath: str) -> str:
    list = os.listdir(datapath)
    dirlist = []

    for name in list:
        dirname = os.path.join(datapath, name)
        if os.path.isdir(dirname):
            dirlist.append(dirname)

    dirlist.sort(key=lambda dirname: os.path.getmtime(dirname))

    if len(dirlist) > 0:
        return os.path.join(datapath, dirlist[-1])

    return ""
