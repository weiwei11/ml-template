# Author: weiwei
from lib.visualizers.base_visualizer import BaseVisualizer


class Visualizer(BaseVisualizer):

    def process_data(self, output, batch):
        classes = list(map(lambda x: x[0], batch['meta']['classes']))
        gt_label = batch['label'].detach().cpu().numpy()[0]
        pred_label = output['pred_label'].detach().cpu().numpy()[0]
        gt_class = classes[gt_label]
        pred_class = classes[pred_label]
        img = batch['img'].detach().cpu().squeeze().numpy()
        return img, gt_class, pred_class

    def draw(self, img, gt_class, pred_class):
        ax = self.fig.add_subplot(1, 1, 1)
        ax.imshow(img)
        ax.set_title('gt: {}, pred: {}'.format(gt_class, pred_class))
        self.fig.show()
