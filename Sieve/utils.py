import re
import os
from typing import List
import numpy as np

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


def mergeOperation(temp_operation_map: dict, operation_map: dict):
    if not operation_map:
        operation_map = temp_operation_map
    else:
        for key, value in temp_operation_map.items():
            if operation_map.get(key) is None:
                operation_map[key] = value
            else:
                for index in operation_map[key].keys():
                    operation_map[key][index].extend(
                        temp_operation_map[key][index])
    return operation_map


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
        if 'embedding' in name:
            # exclude embedding files
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


def generate_save_filepath(name: str, dirname: str = "", is_wechat: bool = False) -> str:
    """
    生成预处理文件的存储路径
    """
    if is_wechat:
        filepath = os.path.join(os.getcwd(), 'data',
                                'preprocessed', 'wechat', dirname, name)
    else:
        filepath = os.path.join(os.getcwd(), 'data',
                                'preprocessed', 'trainticket', dirname, name)

    return filepath

def get_target_label_idx(labels, targets):
    """
        Get the indices of labels that are included in targets.
        :param labels: array of labels
        :param targets: list/tuple of target labels
        :return: list with indices of target labels
        """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()
