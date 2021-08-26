# Kagaya kagaya85@outlook.com
import pandas as pd
import tqdm as tqdm


class item:
    def __init__(self) -> None:
        self.SPAN_ID = 'SpanID'
        self.PARENT_SPAN_ID = 'ParentSpan'
        self.TRACE_ID = 'TraceId'
        self.START_TIME = 'StartTime'
        self.END_TIME = 'EndTime'
        self.SPAN_TYPE = 'SpanType'


ITEM = item()


class span:
    def __init__(self) -> None:
        self.spanId = ''
        self.parentSpanId = ''
        self.traceId = ''
        self.startTime = 0
        self.duration = 0
        self.service = ''
        self.operation = ''
        self.code = 0
        self.isSuccess = False


def load_span(pathList: list):
    """
    load sapn data from pathList
    """
    spansList = []

    for filepath in pathList:
        print(f"load span data from {filepath}")
        spans = pd.read_csv(filepath).drop_duplicates().dropna()
        spansList.append(spans)

    spanData = pd.concat(spansList, axis=0, ignore_index=True)

    return spanData


def build_graph(spanData: dict):
	global ITEM


    for traceId, span in tqdm(spanData.sort_values(by=[ITEM.START_TIME]).groupby([ITEM.TRACE_ID])):
		rootSpan = None
        
        outputGraph = {}

	    yield outputGraph, traceId


def save_data():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
