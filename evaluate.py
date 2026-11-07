# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch


def predict_scores(model, dataset, device, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    S, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            S.append(model(x.to(device)).cpu().numpy())
            Y.append(y.numpy())
    return np.concatenate(S), np.concatenate(Y)


def find_threshold(val_s, val_y):
    """穷举搜索：候选切分点为排序验证得分的相邻中点，取验证准确率最大者"""
    lo, hi = val_s.min(), val_s.max()
    vn = (val_s - lo) / (hi - lo + 1e-12)
    uniq = np.sort(np.unique(vn))
    cands = (uniq[:-1] + uniq[1:]) / 2
    accs = [((vn >= c) == val_y).mean() for c in cands]
    return cands[int(np.argmax(accs))], lo, hi  # 返回归一化阈值与归一化边界


def metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2*tp + fp + fn) else 0.0
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, Accuracy=acc, Precision=prec, Recall=rec, F1=f1)


def main():
    from train import ImgDataset, IMG_H, IMG_W, IN_CH  # 延迟导入，避免循环依赖
    from torchvision import transforms
    from model import DWCN
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--val_dir', required=True)
    ap.add_argument('--test_dir', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([transforms.Resize((IMG_H, IMG_W)), transforms.ToTensor()])
    model = DWCN(IN_CH, IMG_H, IMG_W).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    val_s, val_y = predict_scores(model, ImgDataset(args.val_dir, tf), device)
    tau_n, lo, hi = find_threshold(val_s, val_y)          # 归一化边界仅取自验证集
    tst_s, tst_y = predict_scores(model, ImgDataset(args.test_dir, tf), device)
    tst_n = (tst_s - lo) / (hi - lo + 1e-12)              # 测试集用同一组冻结边界
    y_pred = (tst_n >= tau_n).astype(int)

    r = metrics(tst_y.astype(int), y_pred)
    print(f'normalized threshold = {tau_n:.4f} (val bounds [{lo:.4f}, {hi:.4f}])')
    for k, v in r.items():
        print(f'{k:9s} = {v:.4f}' if isinstance(v, float) else f'{k:9s} = {v}')


if __name__ == '__main__':
    main()
