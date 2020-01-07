import torch
import os
import torch.nn as nn
import time
import shutil
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from tensorboard_logger import configure, log_value
from models.vgg import VGG
from utils.compute_average import AverageMeter


# save model도 만들어야 한다 .
# val 나누는 것도 만들기

class nkModel(object):
    def __init__(self, args, train_loader, val_loader, test_loader):

        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.lr = args.lr
        self.model_name = "nk"
        self.epochs = args.epochs
        self.gamma = args.gamma
        self.best_val_acc = 0
        self.num_train = len(self.train_loader.dataset)
        self.use_tensorboard = args.use_tensorboard
        self.batch_size = args.batch_size
        self.logs_dir = args.logs_dir
        now = datetime.now()  # current date and time
        self.time = now.strftime("%H:%M:%S")
        self.save_dir = './' + args.save_dir  # later change
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + args.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        if args.model_name == 'vgg16':
            # self.model = VGG('VGG16', 0)
            self.model = models.resnet18()
            self.model.fc = nn.Linear(512, 6)

            print(self.model)

            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                             momentum=args.momentum, weight_decay=args.weight_decay)
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=self.gamma, last_epoch=-1)

        # Parallel
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('The number of parametrs of models is', num_params)

        if args.save_load:
            location = args.save_location
            print("locaton", location)
            checkpoint = torch.load(location)
            self.model.load_state_dict(checkpoint['state_dict'])

    def train(self):
        """
        Training Model nk
        """
        self.model.train()

        for epoch in range(self.epochs):
            print('\nEpoch: {}/{} - LR: {:.6f}'.format(epoch + 1, self.epochs,
                                                       self.optimizer.param_groups[0]['lr'], ))
            # Test Accuarcy
            train_losses, train_accs = self.train_one_epoch(epoch)
            self.val(epoch, train_losses, train_accs)

            self.scheduler.step(epoch)

    def train_one_epoch(self, epoch):

        losses = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()
        tic = time.time()

        with tqdm(total=self.num_train) as pbar:
            for i, (inputs, targets, _) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                output = self.model(inputs)
                loss = self.criterion(output, targets)
                # Compute gradient and Do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Measure accuracy and record loss
                prec1 = self.accuracy(output.data, targets)[0]
                losses.update(loss.item(), inputs.size()[0])
                top1.update(prec1.item(), inputs.size()[0])

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model1_loss: {:.3f} - model1_acc: {:.3f}".format(
                            (toc - tic), losses.avg, top1.avg
                        )
                    )
                )

                pbar.update(self.batch_size)

                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + 1

                    log_value('train_loss_%d' % (i + 1), losses.avg, iteration)
                    log_value('train_acc_%d' % (i + 1), top1.avg, iteration)

            return losses, top1

    def val(self, epoch, train_losses, train_accs):

        self.model.eval()

        val_losses = AverageMeter()
        val_accs = AverageMeter()

        for i, (inputs, targets, _) in enumerate(self.val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # forward pass
            outputs = self.model(inputs)

            # cal loss
            loss = self.criterion(outputs, targets)

            # measure accuracy and record loss
            prec = self.accuracy(outputs.data, targets.data, topk=(1,))[0]
            val_losses.update(loss.item(), inputs.size()[0])
            val_accs.update(prec.item(), inputs.size()[0])

        if self.use_tensorboard:
            log_value('valid_loss_%d' % (i + 1), val_losses.avg, epoch + 1)
            log_value('valid_acc_%d' % (i + 1), val_accs.avg, epoch + 1)

        # self.draw_graph(i, epoch, train_losses[i].avg, val_losses[i].avg, train_accs[i].avg, val_accs[i].avg)
        is_best = val_accs.avg > self.best_val_acc
        msg1 = "model_{:d}: train loss: {:.3f} - train acc: {:.3f} "
        msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
        if is_best:
            # self.counter = 0
            msg2 += " [*]"
        msg = msg1 + msg2
        print(msg.format(0 + 1, train_losses.avg, train_accs.avg, val_losses.avg, val_accs.avg))

        self.best_val_acc = max(val_accs.avg, self.best_val_acc)
        self.save_checkpoint(0, {'epoch': epoch + 1, 'model_state': self.model.state_dict(),
                                 'optim_state': self.optimizer.state_dict(),
                                 'best_valid_acc': self.best_val_acc,
                                 }, is_best)

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        # load the best checkpoint
        path = "/tmp/pycharm_project_exam_vision/save/best_1.pth.tar"

        self.load_model(path)
        self.model.eval()
        temp = []
        for i, (inputs, targets, _) in enumerate(self.test_loader):
            inputs = inputs.cuda()
            # forward pass
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            predicted = predicted.numpy()[0]
            temp.append(predicted)

        return temp

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def save_checkpoint(self, i, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.
        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """

        filename = self.model_name + "_" + self.time + "_" + str(i + 1) + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.save_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + "_" + self.time + "_" + str(i + 1) + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.save_dir, filename)
            )

    def load_model(self, path):

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
