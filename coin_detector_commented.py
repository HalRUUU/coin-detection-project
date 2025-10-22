#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬幣偵測與分類系統 (適用台灣 1 / 5 / 10 / 50 元硬幣)
-------------------------------------------------
流程：
1. 使用 HoughCircles 偵測影像中的圓形（硬幣輪廓）
2. 擷取每個圓形區域的特徵（半徑、平均色相、飽和度）
3. 以 KMeans 將特徵分群（最多 4 群，對應 1/5/10/50 元）
4. 根據群心排序及飽和度分配面額
5. 在輸出影像上標註面額並顯示硬幣總數
使用方式：
    python coin_detector_v2.py --image coin_img.jpg
"""

import cv2, argparse, numpy as np
from collections import Counter
from pathlib import Path

def detect_and_classify(image_path, output_prefix="result"):
    # 讀取輸入影像
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    # 取得影像尺寸（高、寬）
    h, w = img_bgr.shape[:2]

    # 轉成灰階並進行中值濾波以減少雜訊
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)

    # ---- Hough 轉換偵測圓形 ----
    minR = max(15, int(min(h, w) * 0.03))   # 最小半徑（依比例）
    maxR = int(min(h, w) * 0.12)            # 最大半徑（依比例）

    circles = cv2.HoughCircles(
        img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(min(h, w) * 0.12),
        param1=100, param2=30, minRadius=minR, maxRadius=maxR
    )

    # 儲存合法的圓座標 (x, y, r)
    detected = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for x, y, r in circles:
            # 過濾掉超出邊界的圓
            if 0 <= x < w and 0 <= y < h and x - r > 0 and y - r > 0 and x + r < w and y + r < h:
                detected.append((x, y, r))

    # ---- 將偵測結果畫出（綠色圓、紅點中心） ----
    drawn = img_bgr.copy()
    for (x, y, r) in detected:
        cv2.circle(drawn, (x, y), r, (0, 255, 0), 3)
        cv2.circle(drawn, (x, y), 2, (0, 0, 255), 3)
    cv2.imwrite(f"{output_prefix}_circles.jpg", drawn)

    # ---- 擷取每個硬幣的顏色特徵 (HSV) ----
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    features = []
    for (x, y, r) in detected:
        Y, X = np.ogrid[:h, :w]
        # 建立圓形遮罩區域
        mask = (X - x) ** 2 + (Y - y) ** 2 <= int(r * 0.9) ** 2
        hsv = img_hsv[mask]
        # 計算平均色相與飽和度
        mean_h, mean_s = float(np.mean(hsv[:, 0])), float(np.mean(hsv[:, 1]))
        features.append([r, mean_h, mean_s])
    features = np.array(features, dtype=np.float32)

    # ---- 使用 KMeans 進行分群 ----
    k = min(4, max(1, len(features)))  # 最多分四群（對應 4 種面額）
    labels = np.zeros(len(features), dtype=int)
    centers = None

    if len(features) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.001)
        _, labels, centers = cv2.kmeans(features, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # ---- 依據半徑大小推測面額 ----
    denoms_present = [1, 5, 10, 50][:k]  # 限制在已知面額數量內
    cluster_to_denom = {}

    if centers is not None:
        order = np.argsort(centers[:, 0])  # 以半徑排序（由小到大）
        for rank, c in enumerate(order):
            cluster_to_denom[c] = denoms_present[rank]

        # 若有四群，進一步用飽和度推估金色 50 元
        if k == 4:
            likely_gold = int(np.argmax(centers[:, 2]))  # 飽和度最高的群
            cluster_to_denom[likely_gold] = 50
            others = [c for c in order if c != likely_gold]
            for c, d in zip(others, [1, 5, 10]):
                cluster_to_denom[c] = d

    # ---- 繪製標籤與統計數量 ----
    counts = Counter()
    labeled = img_bgr.copy()

    for i, (x, y, r) in enumerate(detected):
        lbl = int(labels[i]) if len(features) > 0 else 0
        denom = cluster_to_denom.get(lbl, 0)
        if denom:
            counts[denom] += 1
        # 在硬幣位置標上預測面額
        cv2.putText(labeled, str(denom) if denom else "coin", (x - r, y - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # ---- 顯示總硬幣數量 Overlay ----
    overlay = labeled.copy()
    cv2.rectangle(overlay, (10, 10), (10 + 280, 10 + 64), (0, 0, 0), -1)
    alpha = 0.38
    labeled = cv2.addWeighted(overlay, alpha, labeled, 1 - alpha, 0)
    text = f"Total coins: {len(detected)}"
    cv2.putText(labeled, text, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    # 輸出結果影像
    cv2.imwrite(f"{output_prefix}_labeled.jpg", labeled)

    return len(detected), dict(counts)

if __name__ == "__main__":
    # ---- 命令列參數解析 ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to coin image")
    ap.add_argument("--out", default="result", help="output prefix")
    args = ap.parse_args()

    # 執行主函式
    total, counts = detect_and_classify(args.image, args.out)
    print("Total coins:", total)
    print("Counts:", counts)
