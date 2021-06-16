import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel

from lib.utils.net_utils import save_model, load_model


class Trainer(object):
    def __init__(self, resume, model_dir, save_ep, eval_ep, max_epoch, begin_epoch, use_data_parallel=True):
        self.resume = resume
        self.model_dir = model_dir
        self.eval_ep = save_ep
        self.save_ep = eval_ep
        self.max_epoch = max_epoch
        self.begin_epoch = begin_epoch
        self.use_data_parallel = use_data_parallel

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def train(self, epoch, network, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats = network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(network.parameters(), 40)
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, network, data_loader, evaluator=None, recorder=None):
        network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            with torch.no_grad():
                output, loss, loss_stats, image_stats = network.module(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)

    def fit(self, network, train_loader, val_loader, evaluator, recorder, optimizer, scheduler, network_wrapper=None):
        begin_epoch = load_model(network, optimizer, scheduler, recorder, self.model_dir, resume=self.resume, epoch=self.begin_epoch)

        if network_wrapper is not None:
            network_wrapper = network_wrapper.cuda()
        if self.use_data_parallel:
            network_wrapper = DataParallel(network_wrapper)

        max_epoch = self.max_epoch
        model_dir = self.model_dir
        save_ep = self.save_ep
        eval_ep = self.eval_ep

        for epoch in range(begin_epoch, max_epoch):
            recorder.epoch = epoch
            self.train(epoch, network_wrapper, train_loader, optimizer, recorder)
            scheduler.step()

            if (epoch + 1) % save_ep == 0:
                save_model(network, optimizer, scheduler, recorder, epoch, model_dir)

            if (epoch + 1) % eval_ep == 0:
                self.val(epoch, network_wrapper, val_loader, evaluator, recorder)
