# Author: weiwei

import matplotlib.pyplot as plt


class BaseVisualizer:

    def __init__(self, result_dir=None, figsize=(19.20, 10.80)):
        self.result_dir = result_dir
        self.fig = plt.figure(figsize=figsize)  # used to visualization

    def visualize(self, output, batch):
        all_data = self.process_data(output, batch)

        # visualize
        plt.ion()
        # _, ax = plt.subplots(1, figsize=(19.20, 10.80))
        self.draw(*all_data)
        # plt.show()
        plt.waitforbuttonpress()
        plt.clf()
        # plt.close()

    def visualize_train(self, output, batch):
        # raise NotImplementedError()
        pass

    def process_data(self, output, batch):
        raise NotImplementedError()

    def draw(self, *args):
        raise NotImplementedError()
