import argparse
import os
import logging
from pathlib import Path
import sys
import importlib
import importlib.util
from tqdm import tqdm
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from data_utils.dataloader import SemanticDataset

seg_label_to_cat = {0: 'innerpoint', 1: 'boundarypoint'}

def parse_args():
    parser = argparse.ArgumentParser('Test Model')
    parser.add_argument('--model', type=str, default='SFDNet_cuda',
                        help='Model module name [default: SFDNet_cuda]')
    parser.add_argument('--batch_size', type=int, default=64, help='Chunks per GPU forward [default: 64]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: 0]')
    parser.add_argument('--npoint', type=int, default=500, help='Points per chunk [default: 500]')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root (log directory)')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use additional channels [default: False]')
    parser.add_argument('--eval_metric', type=str, default='f1', help='Which best-checkpoint to load: acc/f1/loss/fpr')
    parser.add_argument('--data_dir', type=str, default='dataset/ABC/noise_none', help='dataset directory for one condition [default: dataset/ABC/noise_none]')
    parser.add_argument('--add_channel', type=int, default=3, help='additional channels')
    parser.add_argument('--save_pc', action='store_true', default=False, help='True if you want to save the predicted results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic chunking [default: 42]')
    return parser.parse_args()

def log_string(logger, message):
    logger.info(message)
    print(message)

def load_model_module(experiment_dir, model_name, logger):
    archived = experiment_dir / (model_name + '.py')
    if archived.exists():
        log_string(logger, 'Loading model from archive: %s' % archived)
        spec = importlib.util.spec_from_file_location(model_name, str(archived))
        module = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = module
        spec.loader.exec_module(module)
        return module
    log_string(logger, 'Loading model from models/: %s' % model_name)
    return importlib.import_module(model_name)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = Path('./log/sem_seg') / args.log_dir
    experiment_dir.mkdir(exist_ok=True, parents=True)
    predicted_results_dir = experiment_dir / 'predicted_results'
    predicted_results_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = experiment_dir / 'eval.txt'
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger, 'PARAMETER ...')
    log_string(logger, args)

    root = args.data_dir

    NUM_CLASSES = 2

    log_string(logger, "Start loading test data ...")
    TEST_DATASET = SemanticDataset(root=root, split='test', npoints=None,
                                   normal_channel=args.normal, add_chn=args.add_channel)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1,
                                                 shuffle=False, num_workers=4)

    log_string(logger, "The number of test samples: %d" % len(TEST_DATASET))

    MODEL = load_model_module(experiment_dir, args.model, logger)

    if args.eval_metric == 'acc':
        checkpoint_path = str(experiment_dir / 'checkpoints' / 'best_acc_model.pth')
        log_string(logger, 'Loading best accuracy model...')
    elif args.eval_metric == 'f1':
        checkpoint_path = str(experiment_dir / 'checkpoints' / 'best_f1_model.pth')
        log_string(logger, 'Loading best F1-score model...')
    elif args.eval_metric == 'loss':
        checkpoint_path = str(experiment_dir / 'checkpoints' / 'best_loss_model.pth')
        log_string(logger, 'Loading best loss model...')
    elif args.eval_metric == 'fpr':
        checkpoint_path = str(experiment_dir / 'checkpoints' / 'best_fpr_model.pth')
        log_string(logger, 'Loading best fpr model...')
    else:
        log_string(logger, 'Invalid eval_metric flag. Using default best model.')
        checkpoint_path = str(experiment_dir / 'checkpoints' / 'model.pth')

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    mk = checkpoint.get('model_kwargs')
    if mk is None:
        mk = dict(normal_channel=args.normal, add_chn=args.add_channel)
    log_string(logger, 'Building model with kwargs: %s' % mk)
    classifier = MODEL.get_model(NUM_CLASSES, **mk).cuda()
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
    pred_count_class = [0 for _ in range(NUM_CLASSES)]

    rng = np.random.default_rng(args.seed)
    _written_names = set()

    with torch.no_grad():
        log_string(logger, '---- TEST EVALUATION (chunked, batched) ----')
        for batch_idx, (points, target) in tqdm(enumerate(testDataLoader),
                                                total=len(testDataLoader), smoothing=0.9):
            pts_np = points[0].numpy()
            tgt_np = target[0].numpy().astype(int)
            N = pts_np.shape[0]
            npoint = args.npoint

            perm = rng.permutation(N)
            num_chunks = int(np.ceil(N / npoint))
            chunk_idx = np.empty((num_chunks, npoint), dtype=np.int64)
            for c in range(num_chunks):
                seg_ids = perm[c * npoint:(c + 1) * npoint]
                if len(seg_ids) < npoint:
                    pad = rng.choice(seg_ids, npoint - len(seg_ids), replace=True)
                    seg_ids = np.concatenate([seg_ids, pad])
                chunk_idx[c] = seg_ids

            chunk_pts = torch.from_numpy(pts_np[chunk_idx]).float().cuda()

            pred_per_point = np.full(N, -1, dtype=np.int64)
            for i in range(0, num_chunks, args.batch_size):
                batch = chunk_pts[i:i + args.batch_size]
                batch = batch.transpose(2, 1)
                seg_pred, _ = classifier(batch)
                pred = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
                for j in range(batch.shape[0]):
                    pred_per_point[chunk_idx[i + j]] = pred[j]

            total_correct += int(np.sum(pred_per_point == tgt_np))
            total_seen += N
            for l in range(NUM_CLASSES):
                total_seen_class[l]     += int(np.sum(tgt_np == l))
                total_correct_class[l]  += int(np.sum((pred_per_point == l) & (tgt_np == l)))
                total_iou_deno_class[l] += int(np.sum((pred_per_point == l) | (tgt_np == l)))
                pred_count_class[l]     += int(np.sum(pred_per_point == l))

            if args.save_pc:
                xyz = pts_np[:, :3]
                pred_result = np.concatenate(
                    [xyz, pred_per_point.reshape(-1, 1), tgt_np.reshape(-1, 1)], axis=1)
                src = TEST_DATASET.datapath[batch_idx]
                out_name = os.path.basename(src)
                if out_name in _written_names:
                    out_name = '%s_%s' % (os.path.basename(os.path.dirname(src)), out_name)
                _written_names.add(out_name)
                save_filename = predicted_results_dir / out_name
                np.savetxt(str(save_filename), pred_result, fmt='%.5f')

    log_string(logger, '---- Detailed Metrics per Class ----')
    per_class = []
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
        per_class.append((acc_l, precision_l, recall_l, fpr_l, f1_l))
        log_string(logger, 'Class %d => Acc=%.4f, Precision=%.4f, Recall=%.4f, FPR=%.4f, F1=%.4f'
                   % (l, acc_l, precision_l, recall_l, fpr_l, f1_l))

    avgs = np.mean(np.array(per_class), axis=0)
    log_string(logger, 'Average Metrics => Acc: %.4f, Precision: %.4f, Recall: %.4f, FPR: %.4f, F1: %.4f'
               % tuple(avgs))


if __name__ == '__main__':
    args = parse_args()
    main(args)
