
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
