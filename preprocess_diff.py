# -*- coding: utf-8 -*-


输出命名（<名>_<标签>.jpg，标签 0=有缺陷差分, 1=无缺陷差分）：
  有缺陷板参与的差分   -> 序号_0.jpg
  仅无缺陷板参与的差分 -> 序号_1.jpg

用法示例：
  python preprocess_diff.py --std 1_1.png --good_dir good --bad_dir bad \
      --out_dir diff_dataset --partners 4 --percentile 97 --roi 0,0,350,280
  
"""
import os
import argparse
import numpy as np
import cv2

# ==================== 论文 2.2 节：几何配准 ====================

class Registrar:
    """ORB + RANSAC 配准器：将待测图配准到标准参考板坐标系"""

    def __init__(self, ref_path, match_ratio=1.0):
        img = cv2.imread(ref_path)
        if img is None:
            raise ValueError(f'无法读取标准参考板: {ref_path}')
        self.ref = img                                  # 参考板（无缺陷标准板）
        self.ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(nfeatures=500)        # 论文：500 个最强响应特征点
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 双向最近邻
        self.match_ratio = match_ratio
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref_gray, None)
        self.cache = {}                                 # path -> (registered|None, info)

    def _ecc_fallback(self, gray):
        """两级兜底第一级：Canny 边缘 + ECC 粗配准（简化实现；论文为 Canny+Hu 矩）"""
        try:
            edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
            ref_e = cv2.Canny(self.ref_gray, 50, 150).astype(np.float32) / 255.0
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
            cc, warp = cv2.findTransformECC(edges, ref_e, warp, cv2.MOTION_EUCLIDEAN,
                                            criteria, None, 5)
            if cc < 0.5:                                # 置信度不足视为失败
                return None
            return cv2.warpAffine(self._cur_img, warp,
                                  (self.ref.shape[1], self.ref.shape[0]),
                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            return None

    def register(self, img_path):
        """返回 (registered_img 或 None, info dict)；None 表示需人工复核"""
        if img_path in self.cache:
            return self.cache[img_path]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'无法读取图像: {img_path}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        info = {'kp': len(kp) if kp is not None else 0, 'inliers': 0, 'rmse': None,
                'fallback': False, 'review': False}
        registered, rmse = self._estimate_and_warp(img, gray, kp, des, info)
        if registered is None:                          # 内点不足 → 两级兜底
            info['fallback'] = True
            self._cur_img = img
            registered = self._ecc_fallback(gray)
            if registered is None:
                info['review'] = True                   # 第二级：标记人工复核
        else:
            info['rmse'] = rmse
            info['inliers'] = int(self._last_inliers)
        self.cache[img_path] = (registered, info)
        return registered, info

    def _estimate_and_warp(self, img, gray, kp, des, info):
        """RANSAC 估计 H（待测图坐标 → 参考图坐标系，论文式(5)方向）并透视变换"""
        if des is None or self.des_ref is None or len(kp) < 4:
            return None, None
        matches = self.matcher.match(des, self.des_ref)          # 双向最近邻
        matches = sorted(matches, key=lambda m: m.distance)
        keep = max(4, int(len(matches) * self.match_ratio))      # 默认保留全部
        matches = matches[:keep]
        if len(matches) < 4:
            return None, None
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self.kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # 论文：Γ=5.0 像素，K=500 次迭代（maxIters=500）
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=500)
        if H is None:
            return None, None
        inliers = mask.ravel().astype(bool)
        n_in = int(inliers.sum())
        self._last_inliers = n_in
        if n_in < 10:                                   # 论文：内点 <10 判定不可靠
            return None, None
        # 论文式(2)：内点重投影误差 RMSE
        proj = cv2.perspectiveTransform(src[inliers], H).reshape(-1, 2)
        obs = dst[inliers].reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((proj - obs) ** 2, axis=1))))
        # 双线性插值透视变换到参考板坐标系（默认 INTER_LINEAR）
        registered = cv2.warpPerspective(img, H, (self.ref.shape[1], self.ref.shape[0]))
        return registered, rmse


# ==================== 论文 2.3 节：差分与噪声抑制 ====================

def threshold_mask(diff, T):
    """论文式(8)：任一通道差分 > T 则该像素三通道保留，否则三通道置零"""
    keep = np.abs(diff).max(axis=2) > T
    return diff * keep[:, :, None]


def remove_small_components(diff, alpha):
    """剔除面积 < alpha 的孤立响应连通域（论文 2.3 节，α=5）"""
    mask = (np.abs(diff).max(axis=2) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < alpha:
            diff[labels == i] = 0
    return diff


def crop_roi(img, roi):
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def save_diff(diff, out_dir, idx, label):
    img = np.clip(diff, 0, 255).astype(np.uint8)
    path = os.path.join(out_dir, f'{idx}_{label}.jpg')
    cv2.imwrite(path, img)
    return path


def list_images(folder):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        raise ValueError(f'{folder} 中没有图像文件')
    return sorted(files)


def pooled_diff(reg_a, reg_b):
    return np.abs(reg_a.astype(np.float32) - reg_b.astype(np.float32)).ravel()


# ==================== 主流程 ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--std', required=True, help='标准参考板图像路径（无缺陷，配准基准）')
    ap.add_argument('--good_dir', required=True, help='无缺陷板图像文件夹')
    ap.add_argument('--bad_dir', required=True, help='有缺陷板图像文件夹')
    ap.add_argument('--out_dir', default='diff_dataset')
    ap.add_argument('--partners', type=int, default=4, help='每张图的交叉差分对象数（3–5）')
    ap.add_argument('--percentile', type=float, default=95.0,
                    help='差分阈值T的百分位（无缺陷样本统计）')
    ap.add_argument('--alpha', type=int, default=5, help='孤立响应连通域面积下限（像素）')
    ap.add_argument('--roi', default=None, help='固定ROI（参考图上 x,y,w,h），如 0,0,350,280')
    ap.add_argument('--match_ratio', type=float, default=1.0,
                    help='保留前 ratio 比例的双向匹配（默认1.0=全部，论文未做比例筛选）')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    assert 1 <= args.partners <= 8
    roi = tuple(int(v) for v in args.roi.split(',')) if args.roi else None

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    reg = Registrar(args.std, match_ratio=args.match_ratio)
    good_paths = [os.path.join(args.good_dir, f) for f in list_images(args.good_dir)]
    bad_paths = [os.path.join(args.bad_dir, f) for f in list_images(args.bad_dir)]
    n_good = len(good_paths)
    if n_good <= args.partners:
        raise ValueError(f'无缺陷图像数量({n_good})需大于交叉差分对象数({args.partners})')

    # ---------- 第一步：配准全部板图 ----------
    print('第1步：ORB+RANSAC 配准 ...')
    review_list, rmses = [], []
    registered = {}
    for tag, paths in [('good', good_paths), ('bad', bad_paths)]:
        for p in paths:
            img, info = reg.register(p)
            registered[p] = (img, info)
            if info['review'] or img is None:
                review_list.append(p)
            elif info['rmse'] is not None:
                rmses.append(info['rmse'])
    rmses = np.array(rmses)
    if len(rmses):
        print(f'配准完成：成功 {len(rmses)} 张，人工复核 {len(review_list)} 张；'
              f'内点RMSE 均值={rmses.mean():.3f} 像素，'
              f'<0.8像素占比={(rmses < 0.8).mean()*100:.1f}%（论文2.2节口径）')
    if review_list:
        with open(os.path.join(args.out_dir, 'review_list.txt'), 'w') as f:
            f.write('\n'.join(review_list))
        print(f'人工复核清单已写入 review_list.txt（{len(review_list)} 张）')

    def reg_roi(p):
        """取配准图（或标准板本身）并按需裁剪ROI"""
        if p == args.std:
            img = reg.ref
        else:
            img = registered[p][0]
        return crop_roi(img, roi) if (roi and img is not None) else img

    usable = lambda p: p == args.std or registered[p][0] is not None

    # ---------- 第二步：T 的统计（无缺陷配对，论文 2.3 节） ----------
    print('第2步：统计差分阈值 T ...')
    good_good = [(good_paths[i], good_paths[(i + k) % n_good])
                 for k in range(1, args.partners + 1) for i in range(n_good)
                 if usable(good_paths[i]) and usable(good_paths[(i + k) % n_good])]
    std_pairs = [(p, args.std) for p in good_paths if usable(p)]
    vals = [pooled_diff(reg_roi(a), reg_roi(b)) for a, b in good_good + std_pairs]
    pooled = np.concatenate(vals)
    T = float(np.percentile(pooled, args.percentile))
    print(f'T = {T:.2f}（无缺陷差分值第 {args.percentile} 百分位）')

    # ---------- 第三步：生成全部差分图 ----------
    tasks = []
    tasks += [(p, args.std, 1) for p in good_paths if usable(p)]
    tasks += [(p, args.std, 0) for p in bad_paths if usable(p)]
    tasks += [(a, b, 1) for a, b in good_good]
    for bp in bad_paths:
        if not usable(bp):
            continue
        for gp in rng.choice([g for g in good_paths if usable(g)],
                             size=args.partners, replace=False):
            tasks.append((bp, gp, 0))

    print(f'第3步：生成 {len(tasks)} 张差分图（序号_0: '
          f'{sum(1 for t in tasks if t[2]==0)}，序号_1: {sum(1 for t in tasks if t[2]==1)}）...')
    idx = 0
    for a_path, b_path, label in tasks:
        diff = np.abs(reg_roi(a_path).astype(np.float32) - reg_roi(b_path).astype(np.float32))
        diff = threshold_mask(diff, T)
        diff = remove_small_components(diff, args.alpha)
        save_diff(diff, args.out_dir, idx, label)
        idx += 1
        if idx % 100 == 0:
            print(f'  已生成 {idx}/{len(tasks)}')
    print(f'完成。输出目录: {args.out_dir}（共 {idx} 张）')


if __name__ == '__main__':
    main()
