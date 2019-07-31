import torch
from torch.autograd import Variable
import time
import os
import sys
import pdb

from utils import AverageMeter, calculate_accuracy, calculate_precision, calculate_recall


def train_epoch(epoch, data_loader, model, criterion_prob, criterion_shift, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_prob = AverageMeter()
    losses_shift = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter() #
    recalls = AverageMeter()

    end_time = time.time()
    # i, (inputs, targets) = next(iter(enumerate(data_loader)))
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            # targets = targets.cuda(async=True)
            target_prob = targets[0].cuda(non_blocking=True)
            target_shift = targets[1].cuda(non_blocking=True)
        inputs = Variable(inputs)
        target_prob = Variable(target_prob)
        target_shift = Variable(target_shift)
        #pdb.set_trace()
        model_outputs = model(inputs)
        prob, shift = model_outputs  # class probabilities and predicted shift
        # print(prob, target_prob)
        # print(shift, target_shift)
        loss_prob = criterion_prob(prob, target_prob)
        loss_shift = criterion_shift(shift, target_shift)
        loss = loss_prob + loss_shift * 1

        acc = calculate_accuracy(prob, target_prob)
        precision = calculate_precision(prob, target_prob) #
        recall = calculate_recall(prob,target_prob)

        losses.update(loss.item(), inputs.size(0))
        losses_prob.update(loss_prob.item(), inputs.size(0))
        losses_shift.update(loss_shift.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        precisions.update(precision, inputs.size(0))
        recalls.update(recall,inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'loss_prob': losses_prob.val,
            'loss_shift': losses_shift.val,
            'acc': accuracies.val,
            'precision':precisions.val,
            'recall':recalls.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}] | lr: {lr:.5f} | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Loss_prob {loss_prob.val:.4f} ({loss_prob.avg:.4f}) | '
                  'Loss_shift {loss_shift.val:.4f} ({loss_shift.avg:.4f}) | '
                  'Acc {acc.val:.3f} ({acc.avg:.3f}) | '
                  'Precision {precision.val:.3f}({precision.avg:.3f}) | '
                  'Recall {recall.val:.3f}({recall.avg:.3f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss_prob=losses_prob,
                      loss_shift=losses_shift,
                      lr=optimizer.param_groups[0]['lr'],
                      acc=accuracies,
                      precision=precisions,
                      recall=recalls))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'loss_prob': losses_prob.val,
        'loss_shift': losses_shift.val,
        'acc': accuracies.avg,
        'precision':precisions.avg,
        'recall':recalls.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
       save_file_path = os.path.join(opt.result_path,
                                     'save_{}.pth'.format(epoch))
       states = {
           'epoch': epoch + 1,
           'arch': opt.arch,
           'state_dict': model.state_dict(),
           'optimizer': optimizer.state_dict(),
       }
       torch.save(states, save_file_path)
