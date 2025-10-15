#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coin detector & classifier (Taiwan 1/5/10/50)
- Detect circles with HoughCircles
- Extract (radius, hue, saturation)
- KMeans (k=4) and map clusters to denominations by size & color
Usage:
    python coin_detector.py --image coin_img.jpg
"""
import cv2, argparse, numpy as np
from collections import Counter

def detect_and_classify(image_path, output_prefix="result"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)

    minR = max(15, int(min(h,w) * 0.03))
    maxR = int(min(h,w) * 0.12)
    circles = cv2.HoughCircles(
        img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(min(h,w)*0.12),
        param1=100, param2=30, minRadius=minR, maxRadius=maxR
    )
    detected = []
    if circles is not None:
        circles = np.round(circles[0,:]).astype(int)
        for x,y,r in circles:
            if 0<=x<w and 0<=y<h and x-r>0 and y-r>0 and x+r<w and y+r<h:
                detected.append((x,y,r))

    drawn = img_bgr.copy()
    for (x,y,r) in detected:
        cv2.circle(drawn, (x,y), r, (0,255,0), 3)
        cv2.circle(drawn, (x,y), 2, (0,0,255), 3)
    cv2.imwrite(f"{output_prefix}_circles.jpg", drawn)

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    features = []
    hgt, wdt = h, w
    for (x,y,r) in detected:
        Y, X = np.ogrid[:hgt,:wdt]
        mask = (X-x)**2 + (Y-y)**2 <= int(r*0.9)**2
        hsv = img_hsv[mask]
        mean_h, mean_s = float(np.mean(hsv[:,0])), float(np.mean(hsv[:,1]))
        features.append([r, mean_h, mean_s])
    features = np.array(features, dtype=np.float32)

    k = min(4, max(1, len(features)))
    labels = np.zeros(len(features), dtype=int)
    centers = None
    if len(features) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.001)
        _, labels, centers = cv2.kmeans(features, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    denoms_present = [1,5,10,50][:k]
    cluster_to_denom = {}
    if centers is not None:
        order = np.argsort(centers[:,0])
        for rank, c in enumerate(order):
            cluster_to_denom[c] = denoms_present[rank]
        if k == 4:
            likely_gold = int(np.argmax(centers[:,2])) # highest saturation
            cluster_to_denom[likely_gold] = 50
            others = [c for c in order if c != likely_gold]
            for c, d in zip(others, [1,5,10]):
                cluster_to_denom[c] = d

    counts = Counter()
    labeled = img_bgr.copy()
    for i, (x,y,r) in enumerate(detected):
        lbl = int(labels[i]) if len(features)>0 else 0
        denom = cluster_to_denom.get(lbl, 0)
        if denom: counts[denom]+=1
        cv2.putText(labeled, str(denom) if denom else "coin", (x-r, y-r-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imwrite(f"{output_prefix}_labeled.jpg", labeled)
    return len(detected), dict(counts)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to coin image")
    ap.add_argument("--out", default="result", help="output prefix")
    args = ap.parse_args()
    total, counts = detect_and_classify(args.image, args.out)
    print("Total coins:", total)
    print("Counts:", counts)
