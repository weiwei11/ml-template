# Author: weiwei


class BaseEvaluator:
    def __init__(self, result_dir):
        self.result_dir = result_dir

    def evaluate(self, output, batch):
        raise NotImplementedError()

    def summarize(self):
        raise NotImplementedError()
