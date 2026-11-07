# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import DWCN, dwcn_loss

IMG_H = IMG_W = 70   # 论文：H = W = 70
IN_CH = 3            # 三通道差分图输入


class ImgDataset(Dataset):
    """差分图数据集：文件名末尾 _0 -> 缺陷(标签1)，_1 -> 无缺陷(标签0)"""
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.items = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.items:
            raise ValueError(f"文件夹 {folder} 中未找到图片文件")
        self.transform = transform

    def __getitem__(self, ix):
        name = self.items[ix]
        m = re.search(r'_(\d+)\.\w+$', name)
        if not m:
            raise ValueError(f"文件名 {name} 不符合 '_数字.扩展名' 约定")
        label = 1 - int(m.group(1))
        img = Image.open(os.path.join(self.folder, name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img.to(torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.items)


@torch.no_grad()
def predict_scores(model, dataset, device, batch_size=64):
    model.eval()
    S, Y = [], []
    for idx in range(0, len(dataset), batch_size):
        xs, ys = [], []
        for j in range(idx, min(idx + batch_size, len(dataset))):
            x, y = dataset[j]
            xs.append(x)
            ys.append(y)
        S.append(model(torch.stack(xs).to(device)).cpu().numpy())
        Y.append(np.array(ys, dtype=np.float32))
    return np.concatenate(S), np.concatenate(Y)


def find_threshold(val_s, val_y):
    """论文 2.1：穷举搜索——候选切分点为排序验证得分的相邻中点，取验证准确率最大者"""
    lo, hi = val_s.min(), val_s.max()
    vn = (val_s - lo) / (hi - lo + 1e-12)
    uniq = np.sort(np.unique(vn))
    cands = (uniq[:-1] + uniq[1:]) / 2
    accs = [((vn >= c) == val_y).mean() for c in cands]
    return cands[int(np.argmax(accs))], lo, hi


def balanced_epoch_indices(dataset, batch_size, rng):
    """每个批次正负样本各半（论文 2.4）"""
    labels = np.array([1 - int(re.search(r'_(\d+)\.\w+$', f).group(1)) for f in dataset.items])
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    half = batch_size // 2
    batches = []
    for b in range(min(len(pos), len(neg)) // half):
        sel = np.concatenate([pos[b * half:(b + 1) * half], neg[b * half:(b + 1) * half]])
        rng.shuffle(sel)
        batches.append(sel)
    return batches


def copy_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True, help='60张训练差分图文件夹（30缺陷+30无缺陷）')
    ap.add_argument('--val_dir', required=True, help='40张验证差分图文件夹（20+20）')
    ap.add_argument('--out', default='dwcn_best.pth', help='最优权重保存路径')
    ap.add_argument('--batch_size', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([transforms.Resize((IMG_H, IMG_W)), transforms.ToTensor()])
    train_ds = ImgDataset(args.train_dir, tf)
    val_ds = ImgDataset(args.val_dir, tf)
    print(f'训练集 {len(train_ds)} 张，验证集 {len(val_ds)} 张，设备 {device}')

    model = DWCN(IN_CH, IMG_H, IMG_W).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)          # 论文：Adam, lr=1e-3
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state, best_tau, best_bounds = -1.0, None, 0.0, (0.0, 1.0)
    for epoch in tqdm(range(1, args.epochs + 1), desc='训练进度'):
        model.train()
        ep_loss, ep_acc, nb = 0.0, 0.0, 0
        tau_run = best_tau if epoch > 10 else None
        for sel in balanced_epoch_indices(train_ds, args.batch_size, rng):
            xs = torch.stack([train_ds[int(i)][0] for i in sel]).to(device)
            ys = torch.stack([train_ds[int(i)][1] for i in sel]).to(device)
            opt.zero_grad()
            S = model(xs)
            loss = dwcn_loss(S, ys, model)
            loss.backward()
            opt.step()
            with torch.no_grad():
                if tau_run is None:
                    pos_s = S[ys == 1]
                    tau_b = pos_s.min() if len(pos_s) > 0 else torch.tensor(0.0, device=device)
                else:
                    lo_, hi_ = best_bounds
                    tau_b = torch.tensor(tau_run * (hi_ - lo_) + lo_, device=device)
                acc = ((S >= tau_b).float() == ys).float().mean()
            ep_loss += loss.item(); ep_acc += acc.item(); nb += 1
        sched.step()

        if epoch % 10 == 0:  # 模型选择：验证集穷举阈值定准确率，保存最优
            val_s, val_y = predict_scores(model, val_ds, device)
            tau_n, lo, hi = find_threshold(val_s, val_y)
            val_acc = ((val_s >= tau_n * (hi - lo) + lo) == val_y).mean()
            print(f'\nEpoch [{epoch}/{args.epochs}] trn_loss={ep_loss/nb:.4f} '
                  f'trn_acc={ep_acc/nb:.4f} | val_acc={val_acc:.4f} (tau_norm={tau_n:.4f})')
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy_state(model)
                best_tau, best_bounds = tau_n, (lo, hi)

    torch.save(best_state, args.out)
    print(f'完成：最优验证准确率 {best_acc:.4f}，权重已保存至 {args.out}')


if __name__ == '__main__':
    main()
