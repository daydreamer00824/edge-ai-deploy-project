import time
from contextlib import contextmanager

class Timerecorder:
    def __init__(self):
        self.records = {}

    def add(self, name:str, cost:float):
        self.records[name] = self.records.get(name, 0.0) + cost

    def get(self, name:str):
        return self.records.get(name, 0.0)
    
    def sum(self, name:str):
        return dict(self.records)
    #如果直接返回 self.records，外部代码拿到的是同一个字典对象的引用。
    # 一旦外部对这个字典进行增删改操作（比如 summary["train"] = 0 或 summary.pop("load_data")），就会直接污染 TimeRecorder 内部的原始记录。
    # 用 dict(self.records) 创建一个浅拷贝（新字典对象），外部对这个字典的任何修改都影响不到原数据，保证了计时数据的完整性


@contextmanager
def time_block(recorder:Timerecorder, name:str):
    start = time.perf_counter()
    try:
        yield
    
    finally:
        end = time.perf_counter()
        recorder.add(name, end - start)