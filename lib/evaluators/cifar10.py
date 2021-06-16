import numpy as np

from lib.evaluators.base_evaluator import BaseEvaluator


class Evaluator(BaseEvaluator):

    def __init__(self, result_dir):
        super().__init__(result_dir)
        self.accuracy = []
        self.class_accuracy = {}

    def evaluate(self, output, batch):
        label_gt = batch['label'].detach().cpu().numpy()[0]
        label_pred = output['pred_label'].detach().cpu().numpy()[0]
        correct = np.int8(label_pred == label_gt)
        self.accuracy.append(correct)  # total accuracy

        # per class
        cur_class = batch['meta']['classes'][label_gt][0]
        if cur_class not in self.class_accuracy.keys():
            self.class_accuracy[cur_class] = [correct]
        else:
            self.class_accuracy[cur_class].append(correct)

    def summarize(self):
        accuracy = np.mean(self.accuracy)
        class_accuracy = {}
        print('total accuracy: {}'.format(accuracy))
        for k, v in self.class_accuracy.items():
            class_accuracy[k] = np.mean(v)
            print('class {} accuracy: {}'.format(k, np.mean(v)))
        self.accuracy = []
        self.class_accuracy = {}
        return {'total_accuracy': accuracy, **class_accuracy}
