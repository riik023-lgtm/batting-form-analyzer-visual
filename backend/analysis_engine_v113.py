from __future__ import annotations


"""
Batting Form Analyzer ver.11.3 ROI adapter

既存analysis_engine.pyの分析ロジックはそのまま使い、
ROIだけを「横位置+幅」から「x/y/width/heightの2D矩形」に拡張する。
これにより縦動画でも、画面上で囲んだ赤枠と同じ領域だけをMediaPipeへ渡せる。
"""

import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import analysis_engine as base

_ROI_BY_PATH: dict[str, tuple[float, float, float, float]] = {}


def _clip01(v):
    return float(np.clip(float(v), 0.0, 1.0))


def _normalize_roi_2d(roi):
    if not roi:
        return 0.0, 0.0, 1.0, 1.0

    width = float(np.clip(roi.get("width", 0.84), 0.12, 1.0))
    height = float(np.clip(roi.get("height", 0.84), 0.18, 1.0))

    x_raw = roi.get("x", None)
    if x_raw is None:
        center = float(np.clip(roi.get("center", 0.5), 0.0, 1.0))
        x0 = center - width / 2.0
    else:
        x0 = float(x_raw)

    y0 = float(roi.get("y", 0.0))
    x0 = float(np.clip(x0, 0.0, max(0.0, 1.0 - width)))
    y0 = float(np.clip(y0, 0.0, max(0.0, 1.0 - height)))
    x1 = min(1.0, x0 + width)
    y1 = min(1.0, y0 + height)

    if x1 - x0 < 0.12:
        x1 = min(1.0, x0 + 0.12)
        x0 = max(0.0, x1 - 0.12)
    if y1 - y0 < 0.18:
        y1 = min(1.0, y0 + 0.18)
        y0 = max(0.0, y1 - 0.18)

    return float(x0), float(y0), float(x1), float(y1)


def _map_crop_to_full_2d(norm, x0, y0, x1, y1):
    mapped = norm.copy()
    mapped[:, 0] = x0 + mapped[:, 0] * max(x1 - x0, 1e-6)
    mapped[:, 1] = y0 + mapped[:, 1] * max(y1 - y0, 1e-6)
    return mapped


def _crop_frame(frame, x0, y0, x1, y1):
    h, w = frame.shape[:2]
    px0 = int(np.clip(round(x0 * w), 0, max(0, w - 2)))
    px1 = int(np.clip(round(x1 * w), px0 + 2, w))
    py0 = int(np.clip(round(y0 * h), 0, max(0, h - 2)))
    py1 = int(np.clip(round(y1 * h), py0 + 2, h))
    crop = frame[py0:py1, px0:px1]
    return crop


def _resize_for_pose(image):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    limit = int(base.MAX_INPUT_WIDTH)
    if max_dim <= limit:
        return image
    scale = limit / max_dim
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _fit_box(x0, y0, x1, y1, frame_w, frame_h):
    # 検出した身体の周りに余白を足す。バットはPoseで検出できないので横余白を多めにする。
    bw = max(x1 - x0, 1e-4)
    bh = max(y1 - y0, 1e-4)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    target_w = max(0.28, bw * 1.80)
    target_h = max(0.48, bh * 1.38)

    # 極端に細長くしない。縦動画でも人物が十分大きくなる範囲に抑える。
    target_w = min(target_w, 0.94)
    target_h = min(target_h, 0.94)

    nx0 = np.clip(cx - target_w / 2, 0, 1 - target_w)
    ny0 = np.clip(cy - target_h / 2, 0, 1 - target_h)
    return {
        "x": float(nx0),
        "y": float(ny0),
        "width": float(target_w),
        "height": float(target_h),
    }


def _candidate_boxes(width, height):
    boxes = [(0.0, 0.0, 1.0, 1.0)]

    # 遠くに映る打者を拡大して探す候補。縦横どちらにも対応。
    boxes += [
        (0.00, 0.00, 0.65, 1.00),
        (0.175, 0.00, 0.65, 1.00),
        (0.35, 0.00, 0.65, 1.00),
        (0.00, 0.00, 1.00, 0.75),
        (0.00, 0.25, 1.00, 0.75),
        (0.00, 0.30, 0.70, 0.70),
        (0.15, 0.30, 0.70, 0.70),
        (0.30, 0.30, 0.70, 0.70),
    ]

    if height > width:
        boxes += [
            (0.05, 0.15, 0.90, 0.80),
            (0.05, 0.28, 0.90, 0.70),
        ]

    # 重複除去
    seen = set()
    unique = []
    for b in boxes:
        key = tuple(round(v, 3) for v in b)
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def suggest_batter_roi(video_path, frame_ratio=0.35):
    frame_rgb = base.get_video_preview_frame(video_path, frame_ratio)
    if frame_rgb is None:
        return {"x": 0.08, "y": 0.08, "width": 0.84, "height": 0.84, "confidence": 0.0}

    h, w = frame_rgb.shape[:2]
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    Options = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = Options(
        base_options=BaseOptions(model_asset_path=str(base.ensure_pose_model())),
        running_mode=RunningMode.IMAGE,
        num_poses=4,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
        output_segmentation_masks=False,
    )

    best = None
    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            for bx, by, bw, bh in _candidate_boxes(w, h):
                x0, y0 = bx, by
                x1, y1 = bx + bw, by + bh
                crop = _crop_frame(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), x0, y0, x1, y1)
                crop = _resize_for_pose(crop)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(image)
                world, norm_crop, _ = base._select_batter_pose(result)
                if world is None or norm_crop is None:
                    continue

                visible = norm_crop[
                    (norm_crop[:, 3] >= 0.30)
                    & np.isfinite(norm_crop[:, 0])
                    & np.isfinite(norm_crop[:, 1])
                ]
                if len(visible) < 6:
                    visible = norm_crop[
                        np.isfinite(norm_crop[:, 0]) & np.isfinite(norm_crop[:, 1])
                    ]
                if len(visible) < 6:
                    continue

                full = _map_crop_to_full_2d(visible, x0, y0, x1, y1)
                fx0 = float(np.clip(np.min(full[:, 0]), 0, 1))
                fx1 = float(np.clip(np.max(full[:, 0]), 0, 1))
                fy0 = float(np.clip(np.min(full[:, 1]), 0, 1))
                fy1 = float(np.clip(np.max(full[:, 1]), 0, 1))

                pose_score = float(base._batter_pose_score(norm_crop))
                span_score = float(
                    (np.max(visible[:, 0]) - np.min(visible[:, 0]))
                    + (np.max(visible[:, 1]) - np.min(visible[:, 1]))
                )
                visibility = float(np.nanmean(visible[:, 3])) if visible.shape[1] > 3 else 0.5
                score = pose_score + 0.9 * span_score + 0.6 * visibility

                if best is None or score > best[0]:
                    best = (score, fx0, fy0, fx1, fy1)

        if best is not None:
            score, x0, y0, x1, y1 = best
            out = _fit_box(x0, y0, x1, y1, w, h)
            out["confidence"] = float(np.clip((score + 1.0) / 7.0, 0.0, 1.0))
            return out
    except Exception:
        pass

    if h > w:
        return {"x": 0.05, "y": 0.12, "width": 0.90, "height": 0.82, "confidence": 0.0}
    return {"x": 0.08, "y": 0.08, "width": 0.84, "height": 0.84, "confidence": 0.0}


def _hand_wrist_alignment_metrics(video_path, frame_number, pose_norm, roi_x0, roi_x1):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {}

    x0, y0, x1, y1 = _ROI_BY_PATH.get(
        str(video_path), (float(roi_x0), 0.0, float(roi_x1), 1.0)
    )
    crop = _crop_frame(frame, x0, y0, x1, y1)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    Options = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = Options(
        base_options=BaseOptions(model_asset_path=str(base.ensure_hand_model())),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.30,
        min_hand_presence_confidence=0.30,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)
    if not result.hand_landmarks:
        return {}

    hands = []
    for hand in result.hand_landmarks:
        arr = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=float)
        arr[:, 0] = x0 + arr[:, 0] * max(x1 - x0, 1e-6)
        arr[:, 1] = y0 + arr[:, 1] * max(y1 - y0, 1e-6)
        hands.append(arr)

    metrics = {}
    used = set()
    frame_h, frame_w = frame.shape[:2]
    aspect = frame_w / max(frame_h, 1)

    for side_name, wrist_idx, elbow_idx in [
        ("left", base.LEFT_WRIST, base.LEFT_ELBOW),
        ("right", base.RIGHT_WRIST, base.RIGHT_ELBOW),
    ]:
        pose_wrist = pose_norm[wrist_idx, :2].copy()
        pose_elbow = pose_norm[elbow_idx, :2].copy()
        pose_wrist[0] *= aspect
        pose_elbow[0] *= aspect

        best_i, best_dist = None, float("inf")
        for i, hand in enumerate(hands):
            if i in used:
                continue
            hand_wrist = hand[0, :2].copy()
            hand_wrist[0] *= aspect
            dist = float(np.linalg.norm(hand_wrist - pose_wrist))
            if dist < best_dist:
                best_i, best_dist = i, dist

        if best_i is None or best_dist > 0.18:
            continue

        used.add(best_i)
        hand = hands[best_i]
        forearm = pose_wrist - pose_elbow
        palm_start = hand[0, :2].copy()
        palm_end = hand[9, :2].copy()
        palm_start[0] *= aspect
        palm_end[0] *= aspect
        palm = palm_end - palm_start
        angle = base._angle_between_2d(forearm, palm)
        if np.isfinite(angle):
            metrics[f"{side_name}_wrist_alignment_proxy_deg"] = float(angle)
            metrics[f"{side_name}_hand_match_distance"] = best_dist
    return metrics


def analyze_video(video_path, batter_side, prefix, batter_roi=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けませんでした: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x0, y0, x1, y1 = _normalize_roi_2d(batter_roi)
    _ROI_BY_PATH[str(video_path)] = (x0, y0, x1, y1)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    Options = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = Options(
        base_options=BaseOptions(model_asset_path=str(base.ensure_pose_model())),
        running_mode=RunningMode.VIDEO,
        num_poses=2 if batter_roi else 4,
        min_pose_detection_confidence=0.30,
        min_pose_presence_confidence=0.30,
        min_tracking_confidence=0.35,
        output_segmentation_masks=False,
    )

    world_list, norm_list, valid_list = [], [], []
    previous_center = None
    previous_norm = None
    reject_run = 0
    frame_no = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            crop = _crop_frame(frame, x0, y0, x1, y1)
            if crop.size == 0:
                raise ValueError("赤枠の範囲を切り抜けませんでした。赤枠を少し広げてください。")

            crop = _resize_for_pose(crop)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(round(frame_no * 1000.0 / fps))
            result = landmarker.detect_for_video(image, timestamp)

            world, norm_crop, center = base._select_batter_pose(result, previous_center)
            raw_found = world is not None and norm_crop is not None
            temporal_ok = raw_found and base._is_temporally_consistent(norm_crop, previous_norm)
            pose_ok = bool(temporal_ok)

            if raw_found and not temporal_ok:
                reject_run += 1
                if reject_run >= 2 and base._batter_pose_score(norm_crop) >= 2.0:
                    pose_ok = True
            elif raw_found:
                reject_run = 0
            else:
                reject_run += 1
                if reject_run >= 3:
                    previous_center = None
                    previous_norm = None

            if pose_ok:
                mapped = _map_crop_to_full_2d(norm_crop, x0, y0, x1, y1)
                world_list.append(world)
                norm_list.append(mapped)
                valid_list.append(True)
                previous_center = center
                previous_norm = norm_crop.copy()
                reject_run = 0
            else:
                world_list.append(np.full((33, 4), np.nan))
                norm_list.append(np.full((33, 4), np.nan))
                valid_list.append(False)

            frame_no += 1

    cap.release()

    if frame_no < 5:
        raise ValueError("動画のフレーム数が少なすぎます。")

    valid = np.asarray(valid_list, dtype=bool)
    valid_count = int(valid.sum())
    longest = base._longest_true_run(valid)

    if valid_count < max(10, int(round(fps * 0.60))) or longest < max(5, int(round(fps * 0.18))):
        raise ValueError(
            "打撃解析に必要な骨格フレームを十分に確保できませんでした。\n"
            f"骨格検出率: {valid.mean():.1%}\n"
            "赤枠の中に頭から足先までが入り、スイング中に手足が枠外へ出ないよう少し余白を広げてください。"
        )

    raw_norm = np.asarray(norm_list, dtype=float)
    world = base.interpolate_landmarks(np.asarray(world_list), valid)
    norm = base.interpolate_landmarks(raw_norm, valid)

    candidates, onset = base.estimate_impact_candidates(world, norm, fps, valid, top_k=3)
    impact = int(candidates[0])
    stance = base.estimate_stance_frame(world, norm, valid, impact, onset, fps)
    q = base._analysis_quality(valid, raw_norm, impact, fps)
    base._assert_analysis_quality(q, fps)

    features, extra = base.calculate_features(
        world, norm, fps, impact, stance, onset, batter_side, valid,
        image_aspect_ratio=(width / max(height, 1)),
    )

    try:
        extra.update(_hand_wrist_alignment_metrics(video_path, impact, norm[impact], x0, x1))
    except Exception:
        pass

    analysis = base.VideoAnalysis(
        path=str(video_path), fps=fps, frame_count=frame_no, width=width, height=height,
        world_landmarks=world, normalized_landmarks=norm, raw_normalized_landmarks=raw_norm,
        valid_mask=valid, impact_frame=impact, stance_frame=stance, swing_onset_frame=onset,
        impact_candidates=[int(v) for v in candidates], features=features, extra_metrics=extra,
        roi_x0=x0, roi_x1=x1, quality_score=float(q["score"]),
        quality_label=str(q["label"]), quality_details=q,
    )
    # dataclassはslotsを使っていないため、ver.11.3用に縦ROIも保持できる。
    analysis.roi_y0 = y0
    analysis.roi_y1 = y1

    base.save_pose_frame_image(
        video_path, stance, norm[stance],
        base.IMPACT_DIR / f"{prefix}_stance_pose.jpg", "STANCE"
    )
    base.save_pose_frame_image(
        video_path, impact, norm[impact],
        base.IMPACT_DIR / f"{prefix}_impact_pose.jpg", "IMPACT"
    )
    return analysis




# ver.11.5: 日本語の練習動画を優先し、海外動画を予備リンクとして保持する。
# 参考動画は研究根拠そのものではなく、改善ドリルの実演例として提示する。
_FOREIGN_PRACTICE_VIDEOS = dict(getattr(base, "PRACTICE_VIDEOS", {}))

_JAPANESE_PRACTICE_VIDEOS = {
    "身体を開くな": {
        "title": "バッティングの『開き』は肩が重要！！最も力が伝わるヘソの前でボールをとらえろ！！",
        "provider": "木村昇吾 実践!野球チャンネル",
        "url": "https://www.youtube.com/watch?v=ExGFWXxTLpc",
        "note": "身体の開きを肩の向きと打点の関係から説明。『開きを抑える』を単純な固定として扱わないための参考。",
    },
    "割れを作れ": {
        "title": "バッティングの『割れ』の体勢の必要性と動作習得のためのドリル・トレーニング",
        "provider": "前田健 野球動作解説TV",
        "url": "https://www.youtube.com/watch?v=quc-2acmK0Y",
        "note": "割れの意味と動作習得ドリルを日本語で詳しく解説。肩―骨盤の捻転差を確認する際の参考。",
    },
    "ボールを引きつけろ": {
        "title": "インサイドアウトが出来ないとバットに力が上手く伝わらない!? 克服する練習法",
        "provider": "木村昇吾 実践!野球チャンネル",
        "url": "https://www.youtube.com/watch?v=3PwGCQxZSWs",
        "note": "手部が身体から離れ過ぎる外回りを見直し、インサイドアウトの軌道を確認するための参考。",
    },
    "軸をぶらすな": {
        "title": "軸がブレやすい人必見！回転速度を上げるおすすめドリル",
        "provider": "Be Baseball Academy / JBS武蔵TV",
        "url": "https://www.youtube.com/watch?v=1qkdStcTACo",
        "note": "回転時の軸ブレを抑えるドリル。横移動をゼロにするのではなく、回転中心を確認する参考。",
    },
    "壁を作れ": {
        "title": "壁と壁の解放を意識したスイング",
        "provider": "野球の未来を創るBASEBALL FUTURE",
        "url": "https://www.youtube.com/watch?v=nvTB5nygB00",
        "note": "インパクトまで前足と体幹で壁を作り、その後に解放する動きを実演。前脚の受け止めの参考。",
    },
    "頭を残せ": {
        "title": "頭が動く選手は要確認！確率を上げるバッティング練習",
        "provider": "Amazing TV",
        "url": "https://www.youtube.com/watch?v=RAJ-r9ClGlo",
        "note": "頭部の大きな移動を抑える練習例。頭を完全固定するのではなく、視線と体幹姿勢を確認する参考。",
    },
    "体重移動をしろ": {
        "title": "体重移動が苦手で打球が弱い…打球スピードを上げる為の必須練習",
        "provider": "Be Baseball Academy / JBS武蔵TV",
        "url": "https://www.youtube.com/watch?v=gRba_eivU-k",
        "note": "支点移動を使った体重移動ドリル。前方移動から回転へのつながりを確認する参考。",
    },
    "腰を使え": {
        "title": "『腰を回せ！』はNG!? 回旋効率を上げる為の練習方法",
        "provider": "Be Baseball Academy / JBS武蔵TV",
        "url": "https://www.youtube.com/watch?v=yRK3bvBUAe8",
        "note": "腰だけを速く回すのではなく、全身の回旋効率を高める練習。骨盤と肩の連動を見る際の参考。",
    },
}

# 元の改善提案関数はPRACTICE_VIDEOSを参照するため、日本語辞書へ差し替える。
base.PRACTICE_VIDEOS = _JAPANESE_PRACTICE_VIDEOS
_original_generate_improvement_advice = base.generate_improvement_advice


def _generate_improvement_advice_japanese_first(score_df, ref, tar):
    df = _original_generate_improvement_advice(score_df, ref, tar)
    if df is None or len(df) == 0:
        return df

    display_to_key = {
        cfg.get("display_name", key): key
        for key, cfg in getattr(base, "FEATURES", {}).items()
    }
    backup_titles = []
    backup_providers = []
    backup_urls = []
    backup_notes = []

    for _, row in df.iterrows():
        key = display_to_key.get(str(row.get("観点", "")), "")
        backup = _FOREIGN_PRACTICE_VIDEOS.get(key, {})
        backup_titles.append(backup.get("title", ""))
        backup_providers.append(backup.get("provider", ""))
        backup_urls.append(backup.get("url", ""))
        backup_notes.append(backup.get("note", ""))

    df = df.copy()
    df["予備動画タイトル"] = backup_titles
    df["予備動画提供元"] = backup_providers
    df["予備動画URL"] = backup_urls
    df["予備動画メモ"] = backup_notes
    return df


base.generate_improvement_advice = _generate_improvement_advice_japanese_first


# 既存エンジンのrun_analysis内部から呼ばれる関数を差し替える。
base.analyze_video = analyze_video
base.suggest_batter_roi = suggest_batter_roi
base._hand_wrist_alignment_metrics = _hand_wrist_alignment_metrics

# main.pyから使う公開API
run_analysis = base.run_analysis
recalculate_with_manual_impacts = base.recalculate_with_manual_impacts
is_measured_frame = base.is_measured_frame

def __getattr__(name):
    return getattr(base, name)
