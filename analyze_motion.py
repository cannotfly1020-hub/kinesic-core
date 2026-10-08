import argparse
import os
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer

"""
【kinesic-core 動作解析エンジン】
COCO-WholeBody (133点) を用いて、足部（踵・母趾先）と骨盤（身体重心近似）を抽出し、
支持基底面（BOS）と垂直重力垂下線を描画する。
"""

def extract_keypoints(predictions):
    if not predictions or len(predictions) == 0:
        return None

    person = predictions[0]
    keypoints = person.get('keypoints', [])
    scores = person.get('keypoint_scores', [])

    if len(keypoints) < 23:
        return None

    pts = np.array(keypoints)
    conf = np.array(scores)

    # 左右股関節（11, 12）の中点をCOM近似
    com = (pts[11] + pts[12]) / 2.0

    # 左足（19:踵, 17:母趾） / 右足（22:踵, 20:母趾）
    return {
        'com': com,
        'l_heel': pts[19], 'l_toe': pts[17], 'l_conf': (conf[19] + conf[17]) / 2.0,
        'r_heel': pts[22], 'r_toe': pts[20], 'r_conf': (conf[22] + conf[20]) / 2.0
    }

def draw_biomechanics(frame, data):
    h, w, _ = frame.shape
    com = data['com']

    # 信頼度が高い方の足（矢状面の手前側）を選択
    if data['l_conf'] >= data['r_conf']:
        heel, toe = data['l_heel'], data['l_toe']
    else:
        heel, toe = data['r_heel'], data['r_toe']

    heel_x, heel_y = int(heel[0]), int(heel[1])
    toe_x, toe_y = int(toe[0]), int(toe[1])
    com_x, com_y = int(com[0]), int(com[1])

    ground_y = max(heel_y, toe_y)
    bos_min_x = min(heel_x, toe_x)
    bos_max_x = max(heel_x, toe_x)

    # 1. 支持基底面（BOS）の床面ライン（エメラルドグリーン）
    cv2.line(frame, (bos_min_x, ground_y), (bos_max_x, ground_y), (0, 220, 130), 4)
    cv2.circle(frame, (heel_x, heel_y), 4, (0, 220, 130), -1)
    cv2.circle(frame, (toe_x, toe_y), 4, (0, 220, 130), -1)

    # 2. 支持基底面の端から真上に立ち上がるガイド壁
    top_y = int(h * 0.08)
    cv2.line(frame, (bos_min_x, ground_y), (bos_min_x, top_y), (0, 180, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (bos_max_x, ground_y), (bos_max_x, top_y), (0, 180, 100), 1, cv2.LINE_AA)

    # 3. 身体重心（COM）から真下に落ちる重力垂下線（水色）
    cv2.line(frame, (com_x, com_y), (com_x, ground_y + 20), (238, 211, 34), 2, cv2.LINE_AA)
    cv2.circle(frame, (com_x, com_y), 6, (68, 63, 244), -1) # 重心点（赤）
    cv2.circle(frame, (com_x, ground_y), 4, (238, 211, 34), -1)

    return frame

def run(input_path, output_path):
    print("[1/3] モデル初期化中 (RTMPose-m WholeBody)...")
    inferencer = MMPoseInferencer(
        pose2d='rtmpose-m_8xb64-270e_coco-wholebody-256x192',
        device='cpu' # GPUがある環境なら 'cuda'
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[2/3] 解析開始: {width}x{height} @ {fps:.1f}fps")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        res = next(inferencer(frame, return_vis=False))
        preds = res['predictions'][0]
        data = extract_keypoints(preds)
        if data:
            frame = draw_biomechanics(frame, data)
        out.write(frame)
        if count % 30 == 0:
            print(f"  {count} フレーム処理完了...")

    cap.release()
    out.release()
    print(f"[3/3] 完了！ 出力先: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="input.mp4", help="入力動画")
    parser.add_argument("--output", "-o", default="output.mp4", help="出力動画")
    args = parser.parse_args()
    run(args.input, args.output)
