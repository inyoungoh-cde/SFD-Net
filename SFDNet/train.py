import argparse
import os
from data_utils.dataloader import SemanticDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_label_to_cat = {0: 'innerpoint', 1: 'boundarypoint'}

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='SFDNet_cuda', help='model name [default: SFDNet_cuda]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=500, help='Point Number [default: 500]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--data_dir', type=str, default='dataset/ABC/noise_none', help='dataset directory for one condition [default: dataset/ABC/noise_none]')
    parser.add_argument('--add_channel', type=int, default=3, help='additional channels')
    parser.add_argument('--ftl_alpha', type=float, default=0.8, help='FocalTversky alpha (FP weight)')
    parser.add_argument('--ftl_beta' , type=float, default=0.2, help='FocalTversky beta (FN weight)')
    parser.add_argument('--ftl_gamma', type=float, default=1.5, help='FocalTversky gamma (focusing param)')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.data_dir
    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    log_string("Start loading training data ...")
    TRAIN_DATASET = SemanticDataset(root=root, split='train', npoints=NUM_POINT, normal_channel=args.normal, add_chn=args.add_channel)
    log_string("Start loading test data ...")
    TEST_DATASET = SemanticDataset(root=root, split='val', npoints=NUM_POINT, normal_channel=args.normal, add_chn=args.add_channel)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    for util in ('pointnet2_utils.py', 'pointnet2_utils_cuda.py'):
        if os.path.exists('models/%s' % util):
            shutil.copy('models/%s' % util, str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES, normal_channel=args.normal, add_chn=args.add_channel).cuda()
    criterion = MODEL.get_loss(alpha=args.ftl_alpha, beta=args.ftl_beta, gamma=args.ftl_gamma).cuda()
    classifier.apply(inplace_relu)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=LEARNING_RATE_CLIP)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum


    global_epoch = 0
    best_f1 = 0
    best_acc = 0
    lowest_fpr= float('inf')
    lowest_loss= float('inf')

    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        classifier = classifier.train()

        total_seen_class = np.zeros(NUM_CLASSES)
        total_correct_class = np.zeros(NUM_CLASSES)
        pred_count_class = np.zeros(NUM_CLASSES)

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)


            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]


            loss = criterion(seg_pred, target, trans_feat, weights)

            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / float(len(trainDataLoader)))

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum(batch_label == l)
                total_correct_class[l] += np.sum((pred_choice  == l) & (batch_label == l))
                pred_count_class[l] += np.sum(pred_choice  == l)

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        part_accs = np.zeros(NUM_CLASSES)
        part_precisions = np.zeros(NUM_CLASSES)
        part_recalls = np.zeros(NUM_CLASSES)
        part_fprs = np.zeros(NUM_CLASSES)
        part_f1s = np.zeros(NUM_CLASSES)

        for l in range(NUM_CLASSES):
            TP = total_correct_class[l]
            FN = total_seen_class[l] - TP
            FP = pred_count_class[l] - TP
            TN = total_seen - (TP + FP + FN)

            acc_l = (TP + TN) / (total_seen + 1e-6)
            precision_l = TP / (TP + FP + 1e-6)
            recall_l = TP / (TP + FN + 1e-6)
            fpr_l = FP / (FP + TN + 1e-6)
            f1_l = 2 * precision_l * recall_l / (precision_l + recall_l + 1e-6)

            part_accs[l] = acc_l
            part_precisions[l] = precision_l
            part_recalls[l] = recall_l
            part_fprs[l] = fpr_l
            part_f1s[l] = f1_l

            log_string('Train - Part %d => Acc=%.4f, Precision=%.4f, Recall=%.4f, FPR=%.4f, F1=%.4f' %
                       (l, acc_l, precision_l, recall_l, fpr_l, f1_l))
        avg_train_acc = np.mean(part_accs)
        avg_train_f1 = np.mean(part_f1s)
        log_string('Train - Average Accuracy: %.4f, Average F1: %.4f' % (avg_train_acc, avg_train_f1))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')


        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)

            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            pred_count_class = [0 for _ in range(NUM_CLASSES)]

            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)


                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()

                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]


                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                    pred_count_class[l] += np.sum(pred_val == l)

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            raw_counts = total_seen_class
            total_samples = raw_counts[0] + raw_counts[1] + 1e-6
            ratio0 = raw_counts[0] / total_samples
            ratio1 = raw_counts[1] / total_samples
            log_string(f"Validation raw counts: class0={raw_counts[0]}, class1={raw_counts[1]}")
            log_string(f"Validation observed ratios: class0={ratio0:.4f}, class1={ratio1:.4f}")
            if raw_counts[1] == 0:
                log_string("WARNING: No class-1 samples were seen in validation this epoch!")
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log_string('Validating mean loss: %f' % (loss_sum / num_batches))
            now_loss=(loss_sum / num_batches)

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)

            part_accs = np.zeros(NUM_CLASSES)
            part_precisions = np.zeros(NUM_CLASSES)
            part_recalls = np.zeros(NUM_CLASSES)
            part_fprs = np.zeros(NUM_CLASSES)
            part_f1s = np.zeros(NUM_CLASSES)

            for l in range(NUM_CLASSES):
                TP = total_correct_class[l]
                FN = total_seen_class[l] - TP
                FP = pred_count_class[l] - TP
                TN = total_seen - (TP + FP + FN)

                acc_l = (TP + TN) / (total_seen + 1e-6)
                precision_l = TP / (TP + FP + 1e-6)
                recall_l = TP / (TP + FN + 1e-6)
                fpr_l = FP / (FP + TN + 1e-6)
                f1_l = 2 * precision_l * recall_l / (precision_l + recall_l + 1e-6)

                part_accs[l] = acc_l
                part_precisions[l] = precision_l
                part_recalls[l] = recall_l
                part_fprs[l] = fpr_l
                part_f1s[l] = f1_l

                log_string('Part %d => Acc=%.4f, Precision=%.4f, Recall=%.4f, FPR=%.4f, F1=%.4f'
                           % (l, acc_l, precision_l, recall_l, fpr_l, f1_l))

            avg_part_acc = np.mean(part_accs)
            avg_part_f1 = np.mean(part_f1s)
            avg_part_fpr = np.mean(part_fprs)

            if avg_part_acc >= best_acc:
                best_acc = avg_part_acc
                logger.info('Save best accuracy model...')
                savepath = str(checkpoints_dir) + '/best_acc_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'avg_part_acc': avg_part_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            if avg_part_f1 >= best_f1:
                best_f1 = avg_part_f1
                logger.info('Save best F1-score model...')
                savepath = str(checkpoints_dir) + '/best_f1_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'avg_part_f1': avg_part_f1,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            if avg_part_fpr < lowest_fpr:
                lowest_fpr = avg_part_fpr
                logger.info('Save Lowest FPR model...')
                savepath = str(checkpoints_dir) + '/best_fpr_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'avg_part_fpr': avg_part_fpr,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            if now_loss < lowest_loss:
                lowest_loss = now_loss
                logger.info('Save Lowest loss model...')
                savepath = str(checkpoints_dir) + '/best_loss_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'avg_part_loss': now_loss,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            log_string('Best accuracy and f1score: %f and %f ' % (best_acc, best_f1))
            log_string('Lowest fpr and loss: %f and %f ' % (avg_part_fpr, now_loss))
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
