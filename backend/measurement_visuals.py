from __future__ import annotations

import base64
import math
from typing import Any

import cv2
import numpy as np

import analysis_engine_v113 as engine


FEATURE_META = {
    "身体を開くな": {
        "display": "身体の開き",
        "how": "構えとインパクトで肩のラインがどれだけ回ったかを比較します。",
        "note": "STANCE / IMPACT の肩ラインを強調",
    },
    "割れを作れ": {
        "display": "割れ",
        "how": "インパクト前の肩ラインと骨盤ラインの相対的なひねり差を見ます。",
        "note": "肩ラインと骨盤ラインを同一フレームで表示",
    },
    "ボールを引きつけろ": {
        "display": "ボールの引きつけ",
        "how": "インパクト付近の両手中心と体幹中心の距離を肩幅で正規化して比べます。",
        "note": "手の中心―体幹中心を線で表示",
    },
    "軸をぶらすな": {
        "display": "軸のブレ",
        "how": "構えからインパクトまでの体幹中心の横方向の移動幅を見ます。",
        "note": "構え / インパクトの体幹中心を強調",
    },
    "壁を作れ": {
        "display": "前脚の壁",
        "how": "インパクト時の前脚の股関節・膝・足首から膝角度を計測します。",
        "note": "前脚3点と膝角度を表示",
    },
    "頭を残せ": {
        "display": "頭の残り",
        "how": "構えからインパクトまでの頭部位置の移動量を肩幅で正規化して見ます。",
        "note": "構え / インパクトの頭部位置を強調",
    },
    "体重移動をしろ": {
        "display": "体重移動",
        "how": "骨盤中心の水平移動を体重移動のproxyとして比較します。真の重心や荷重は直接測定していません。",
        "note": "構え / インパクトの骨盤中心を強調",
    },
    "腰を使え": {
        "display": "腰の回転",
        "how": "インパクト前後の骨盤ラインの回転速度を動画フレーム間の変化から求めます。",
        "note": "骨盤回転速度ピーク周辺の3コマを表示",
    },
}


PAIRS = [
    (engine.LEFT_SHOULDER, engine.RIGHT_SHOULDER),
    (engine.LEFT_HIP, engine.RIGHT_HIP),
    (engine.LEFT_SHOULDER, engine.LEFT_ELBOW),
    (engine.LEFT_ELBOW, engine.LEFT_WRIST),
    (engine.RIGHT_SHOULDER, engine.RIGHT_ELBOW),
    (engine.RIGHT_ELBOW, engine.RIGHT_WRIST),
    (engine.LEFT_SHOULDER, engine.LEFT_HIP),
    (engine.RIGHT_SHOULDER, engine.RIGHT_HIP),
    (engine.LEFT_HIP, engine.LEFT_KNEE),
    (engine.LEFT_KNEE, engine.LEFT_ANKLE),
    (engine.RIGHT_HIP, engine.RIGHT_KNEE),
    (engine.RIGHT_KNEE, engine.RIGHT_ANKLE),
]


ORANGE = (0, 155, 255)
CYAN = (255, 210, 40)
GREEN = (80, 220, 120)
WHITE = (245, 245, 245)
GRAY = (150, 165, 160)
RED = (70, 70, 245)


def _roi(analysis):
    x0 = float(getattr(analysis, "roi_x0", 0.0))
    x1 = float(getattr(analysis, "roi_x1", 1.0))
    y0 = float(getattr(analysis, "roi_y0", 0.0))
    y1 = float(getattr(analysis, "roi_y1", 1.0))
    if not (0 <= x0 < x1 <= 1):
        x0, x1 = 0.0, 1.0
    if not (0 <= y0 < y1 <= 1):
        y0, y1 = 0.0, 1.0
    return x0, y0, x1, y1


def _read_frame(analysis, frame_number: int):
    frame_number = int(np.clip(frame_number, 0, max(0, analysis.frame_count - 1)))
    cap = cv2.VideoCapture(str(analysis.path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, None, frame_number

    x0, y0, x1, y1 = _roi(analysis)
    h, w = frame.shape[:2]
    px0, px1 = int(round(x0 * w)), int(round(x1 * w))
    py0, py1 = int(round(y0 * h)), int(round(y1 * h))
    px0 = int(np.clip(px0, 0, max(0, w - 2)))
    px1 = int(np.clip(px1, px0 + 2, w))
    py0 = int(np.clip(py0, 0, max(0, h - 2)))
    py1 = int(np.clip(py1, py0 + 2, h))
    crop = frame[py0:py1, px0:px1].copy()
    norm = np.asarray(analysis.normalized_landmarks[frame_number], dtype=float).copy()
    norm[:, 0] = (norm[:, 0] - x0) / max(x1 - x0, 1e-6)
    norm[:, 1] = (norm[:, 1] - y0) / max(y1 - y0, 1e-6)
    return crop, norm, frame_number


def _pt(norm, idx, w, h):
    if idx >= len(norm) or not np.all(np.isfinite(norm[idx, :2])):
        return None
    x = int(round(float(norm[idx, 0]) * w))
    y = int(round(float(norm[idx, 1]) * h))
    return (x, y)


def _mid(a, b):
    if a is None or b is None:
        return None
    return (int(round((a[0] + b[0]) / 2)), int(round((a[1] + b[1]) / 2)))


def _draw_skeleton(frame, norm, alpha=0.55):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    for a, b in PAIRS:
        pa, pb = _pt(norm, a, w, h), _pt(norm, b, w, h)
        if pa and pb:
            cv2.line(overlay, pa, pb, GRAY, 2, cv2.LINE_AA)
    for idx in {engine.NOSE, engine.LEFT_SHOULDER, engine.RIGHT_SHOULDER, engine.LEFT_HIP, engine.RIGHT_HIP,
                engine.LEFT_KNEE, engine.RIGHT_KNEE, engine.LEFT_ANKLE, engine.RIGHT_ANKLE,
                engine.LEFT_WRIST, engine.RIGHT_WRIST}:
        p = _pt(norm, idx, w, h)
        if p:
            cv2.circle(overlay, p, 3, WHITE, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)


def _put_badge(frame, text, x=12, y=28):
    cv2.rectangle(frame, (x-6, y-21), (x + max(92, len(text)*10), y+7), (22, 28, 26), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2, cv2.LINE_AA)


def _line(frame, a, b, color=ORANGE, width=5):
    if a and b:
        cv2.line(frame, a, b, color, width, cv2.LINE_AA)
        cv2.circle(frame, a, max(4, width), color, -1, cv2.LINE_AA)
        cv2.circle(frame, b, max(4, width), color, -1, cv2.LINE_AA)


def _arrow(frame, a, b, color=ORANGE, width=4):
    if a and b:
        cv2.arrowedLine(frame, a, b, color, width, cv2.LINE_AA, tipLength=0.16)


def _encode(frame, max_width=680, quality=74):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, max(1, int(round(h*scale)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def _pad_height(frame, height):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if h == height:
        return frame
    scale = height / h
    return cv2.resize(frame, (max(1, int(round(w*scale))), height), interpolation=cv2.INTER_AREA)


def _panels(frames, labels):
    good = [(f, l) for f, l in zip(frames, labels) if f is not None]
    if not good:
        return None
    target_h = min(520, max(f.shape[0] for f, _ in good))
    panels = []
    for f, label in good:
        p = _pad_height(f, target_h)
        _put_badge(p, label)
        panels.append(p)
    gap = np.full((target_h, 8, 3), 235, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, gap, p])
    return out


def _draw_shoulders(frame, norm, color=ORANGE):
    h, w = frame.shape[:2]
    _line(frame, _pt(norm, engine.LEFT_SHOULDER, w, h), _pt(norm, engine.RIGHT_SHOULDER, w, h), color, 5)


def _draw_hips(frame, norm, color=CYAN):
    h, w = frame.shape[:2]
    _line(frame, _pt(norm, engine.LEFT_HIP, w, h), _pt(norm, engine.RIGHT_HIP, w, h), color, 5)


def _separation_frame(analysis):
    try:
        world = np.asarray(analysis.world_landmarks, dtype=float)
        sh = np.array([engine.horizontal_line_angle(f[engine.LEFT_SHOULDER], f[engine.RIGHT_SHOULDER]) for f in world])
        hp = np.array([engine.horizontal_line_angle(f[engine.LEFT_HIP], f[engine.RIGHT_HIP]) for f in world])
        sep = engine.circular_angle_difference(sh, hp).astype(float)
        start = max(int(analysis.stance_frame), int(analysis.impact_frame - round(0.7 * analysis.fps)))
        end = int(analysis.impact_frame) + 1
        idxs = np.arange(start, end)
        valid = np.asarray(analysis.valid_mask[start:end], dtype=bool)
        vals = sep[start:end]
        idxs = idxs[valid & np.isfinite(vals)]
        vals = vals[valid & np.isfinite(vals)]
        if len(idxs):
            target = np.nanpercentile(vals, 90)
            return int(idxs[np.nanargmin(np.abs(vals-target))])
    except Exception:
        pass
    return int(analysis.impact_frame)


def _draw_body_opening(analysis):
    f1, n1, _ = _read_frame(analysis, analysis.stance_frame)
    f2, n2, _ = _read_frame(analysis, analysis.impact_frame)
    if f1 is not None:
        f1 = _draw_skeleton(f1, n1); _draw_shoulders(f1, n1)
    if f2 is not None:
        f2 = _draw_skeleton(f2, n2); _draw_shoulders(f2, n2)
    return _panels([f1, f2], ['STANCE', 'IMPACT'])


def _draw_separation(analysis):
    frame_no = _separation_frame(analysis)
    f, n, _ = _read_frame(analysis, frame_no)
    if f is None: return None
    f = _draw_skeleton(f, n)
    _draw_shoulders(f, n, ORANGE); _draw_hips(f, n, CYAN)
    _put_badge(f, 'MAX SEPARATION')
    return f


def _draw_hand_torso(analysis):
    f, n, _ = _read_frame(analysis, analysis.impact_frame)
    if f is None: return None
    f = _draw_skeleton(f, n)
    h, w = f.shape[:2]
    lw, rw = _pt(n, engine.LEFT_WRIST, w, h), _pt(n, engine.RIGHT_WRIST, w, h)
    ls, rs = _pt(n, engine.LEFT_SHOULDER, w, h), _pt(n, engine.RIGHT_SHOULDER, w, h)
    lh, rh = _pt(n, engine.LEFT_HIP, w, h), _pt(n, engine.RIGHT_HIP, w, h)
    hands = _mid(lw, rw); torso = _mid(_mid(ls, rs), _mid(lh, rh))
    _line(f, hands, torso, ORANGE, 5); _put_badge(f, 'HAND - TORSO')
    return f


def _draw_axis(analysis):
    frames=[]
    for frame_no in [analysis.stance_frame, analysis.impact_frame]:
        f,n,_=_read_frame(analysis, frame_no)
        if f is None: frames.append(None); continue
        f=_draw_skeleton(f,n); h,w=f.shape[:2]
        ls,rs=_pt(n,engine.LEFT_SHOULDER,w,h),_pt(n,engine.RIGHT_SHOULDER,w,h)
        lh,rh=_pt(n,engine.LEFT_HIP,w,h),_pt(n,engine.RIGHT_HIP,w,h)
        c=_mid(_mid(ls,rs),_mid(lh,rh))
        if c:
            cv2.circle(f,c,10,ORANGE,-1,cv2.LINE_AA)
            cv2.line(f,(c[0],max(0,c[1]-90)),(c[0],min(h-1,c[1]+90)),ORANGE,3,cv2.LINE_AA)
        frames.append(f)
    return _panels(frames,['STANCE','IMPACT'])


def _angle_deg(a,b,c):
    if not a or not b or not c: return None
    va=np.array(a,float)-np.array(b,float); vc=np.array(c,float)-np.array(b,float)
    den=np.linalg.norm(va)*np.linalg.norm(vc)
    if den < 1e-6: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(va,vc)/den,-1,1))))


def _draw_knee(analysis,batter_side):
    f,n,_=_read_frame(analysis,analysis.impact_frame)
    if f is None:return None
    f=_draw_skeleton(f,n); h,w=f.shape[:2]
    hip,knee,ankle=(engine.LEFT_HIP,engine.LEFT_KNEE,engine.LEFT_ANKLE) if batter_side=='右打者' else (engine.RIGHT_HIP,engine.RIGHT_KNEE,engine.RIGHT_ANKLE)
    a,b,c=_pt(n,hip,w,h),_pt(n,knee,w,h),_pt(n,ankle,w,h)
    _line(f,a,b,ORANGE,6);_line(f,b,c,ORANGE,6)
    ang=_angle_deg(a,b,c)
    if b and ang is not None:
        cv2.putText(f,f'{ang:.0f} deg',(b[0]+12,max(25,b[1]-12)),cv2.FONT_HERSHEY_SIMPLEX,0.68,WHITE,2,cv2.LINE_AA)
    _put_badge(f,'FRONT KNEE')
    return f


def _draw_head(analysis):
    frames=[]
    for frame_no in [analysis.stance_frame,analysis.impact_frame]:
        f,n,_=_read_frame(analysis,frame_no)
        if f is None:frames.append(None);continue
        f=_draw_skeleton(f,n);h,w=f.shape[:2];p=_pt(n,engine.NOSE,w,h)
        if p:
            cv2.circle(f,p,13,ORANGE,4,cv2.LINE_AA)
            cv2.line(f,(0,p[1]),(w-1,p[1]),ORANGE,2,cv2.LINE_AA)
        frames.append(f)
    return _panels(frames,['STANCE','IMPACT'])


def _draw_weight(analysis):
    frames=[]
    for frame_no in [analysis.stance_frame,analysis.impact_frame]:
        f,n,_=_read_frame(analysis,frame_no)
        if f is None:frames.append(None);continue
        f=_draw_skeleton(f,n);h,w=f.shape[:2]
        lh,rh=_pt(n,engine.LEFT_HIP,w,h),_pt(n,engine.RIGHT_HIP,w,h); c=_mid(lh,rh)
        if c: cv2.circle(f,c,11,ORANGE,-1,cv2.LINE_AA)
        _draw_hips(f,n,ORANGE)
        frames.append(f)
    return _panels(frames,['STANCE','IMPACT'])


def _draw_hip_rotation(analysis):
    peak=int(round(float(analysis.extra_metrics.get('hip_peak_frame',analysis.impact_frame))))
    step=max(1,int(round(0.05*analysis.fps)))
    nums=[max(0,peak-step),peak,min(analysis.frame_count-1,peak+step)]
    frames=[]
    for frame_no in nums:
        f,n,_=_read_frame(analysis,frame_no)
        if f is None:frames.append(None);continue
        f=_draw_skeleton(f,n);_draw_hips(f,n,ORANGE);frames.append(f)
    return _panels(frames,['BEFORE','PEAK','AFTER'])


def _visual_for(name, analysis, batter_side):
    if name=='身体を開くな': return _draw_body_opening(analysis)
    if name=='割れを作れ': return _draw_separation(analysis)
    if name=='ボールを引きつけろ': return _draw_hand_torso(analysis)
    if name=='軸をぶらすな': return _draw_axis(analysis)
    if name=='壁を作れ': return _draw_knee(analysis,batter_side)
    if name=='頭を残せ': return _draw_head(analysis)
    if name=='体重移動をしろ': return _draw_weight(analysis)
    if name=='腰を使え': return _draw_hip_rotation(analysis)
    return None


def build_measurement_visuals(reference_analysis, target_analysis, batter_side: str):
    out=[]
    for key in engine.FEATURES.keys():
        meta=FEATURE_META[key]
        try: ref=_encode(_visual_for(key,reference_analysis,batter_side))
        except Exception: ref=None
        try: tar=_encode(_visual_for(key,target_analysis,batter_side))
        except Exception: tar=None
        out.append({
            'key':key,
            'display_name':meta['display'],
            'how_measured':meta['how'],
            'image_note':meta['note'],
            'reference_image':ref,
            'target_image':tar,
        })
    return out
