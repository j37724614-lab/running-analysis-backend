"""
Current version: 5 cameras for 100 meter. (Fixed)
Import module → Setting → Tracking (YOLOv11m-BoT-SORT) → Pose Estimation (YOLOv11L-pose)
 → Edge checking → 


Part I: Import Modules
Part II: Define functions
    II-I: Video reformatting (Optional)
    II-II: Tracking
    II-III: Pose Estimation (Skelton)
    II-IV: 
    II-V: 
Part III: Generate Final Video
Part IV: __main__ & Setting

Edit: Part IV
"""
#########################
# Part I: Import Modules
#########################
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from IPython.display import display
import matplotlib.pyplot as plt
import csv, json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter, find_peaks
from time import time

# video setting
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv")  # 可根據需要修改
IN_VIDEO_EXTENSION = ".mov"
VIDEO_PATTERN = "cam{}" + IN_VIDEO_EXTENSION  # 例如 video1.mp4 ~ video5.mp4；用來開啟影片(合併用)
TO_FPS = 60
TO_HEIGHT = 1080

# tracking setting
WEIGHTS_DET = "yolo11m.pt"   # ultralytics model name (auto-downloads/caches locally)
WEIGHTS_POSE = "yolo11l-pose.pt"   # ultralytics model name (auto-downloads/caches locally)
DEVICE = "cuda:0"
MAX_WORKERS_DET = 4   # Num of cores for tracking - Multiple processing
MAX_WORKERS_POSE = 2   # Num of cores for pose eatimation - Multiple processing
CONF = 0.20     # Confidence for Detection (Tracking) & Pose estimation
IOU = 0.50      # IOU for Detection (Tracking) & Pose estimation
CLASSES = [0]   # 0 = person (COCO)  # for Detection (tracking)

# COCO 17關鍵點名稱 & 骨架連線（Ultralytics/COCO 順序）
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# draw graph setting
LABEL_FONTSIZE = 20
TICK_FONTSIZE  = 8
FONT_NAME      = "Calibri light"   # 或 "Microsoft JhengHei"

#########################
# Part II: Define functions
#########################

# Get Edge & FPS
########################
def get_edge_fps(file_path, camera_no, L, R, C):
    original_cap = cv2.VideoCapture(file_path)
    width = original_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    return [camera_no, L/width*1920, R/width*1920, C/width*1920, 2000/(R/width*1920-L/width*1920), 60, 1/60]


# Reformatting videos (to 1080P, 60FPS)
########################
def _convert_single_video(raw_folder_path, filename, to_height, to_FPS, edited_folder):
    """單支影片轉檔（先試 GPU NVENC，失敗就 fallback 到 CPU libx264）"""
    input_path = os.path.join(raw_folder_path, filename)
    output_path = os.path.join(
        edited_folder,
        os.path.splitext(filename)[0] + ".mov"
    )
    
    if os.path.exists(output_path):
        print(f'File is exists: {output_path}')
        return filename
    else:
        print(f"開始處理：{filename}")

        # ====== 版本 A：GPU (NVENC) 嘗試 ======
        cmd_nvenc = [
            "ffmpeg",
            "-y",
            # 先暫時拿掉 -hwaccel cuda，讓 ffmpeg 自己決定
            # 如果你確定沒問題再加回來：
            # "-hwaccel", "cuda",
            "-i", input_path,

            "-vf", f"scale=-1:{to_height}",
            "-r", str(to_FPS),

            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "19",
            "-b:v", "5M",

            "-c:a", "aac",
            "-b:a", "192k",

            output_path,
        ]

        try:
            print(f"  ➤ 嘗試使用 GPU (h264_nvenc) 轉檔...")
            # capture_output=True 才看得到 ffmpeg 的 stderr
            result = subprocess.run(
                cmd_nvenc,
                check=True,
                capture_output=True,
                text=True
            )
            # 如果成功，你可以視需要印出 result.stdout
            print(f"  ✅ GPU 轉檔成功：{output_path}")
            return filename

        except subprocess.CalledProcessError as e:
            print(f"  ❌ GPU 轉檔失敗（h264_nvenc）for {filename}")
            print("  --- ffmpeg stderr 開始 ---")
            print(e.stderr)
            print("  --- ffmpeg stderr 結束 ---")
            print("  → 改用 CPU (libx264) 再試一次...\n")

        # ====== 版本 B：CPU libx264 fallback ======
        cmd_cpu = [
            "ffmpeg",
            "-y",
            "-i", input_path,

            "-vf", f"scale=-1:{to_height}",
            "-r", str(to_FPS),

            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",

            "-c:a", "aac",
            "-b:a", "192k",

            output_path,
        ]

        # 這裡如果還失敗，才真的 raise 出去給外層
        result_cpu = subprocess.run(
            cmd_cpu,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✅ CPU 轉檔成功：{output_path}")
        return filename

def reformatting_videos(raw_folder_path, max_workers=None):
    # 取得檔案清單
    video_files = [
        f for f in os.listdir(raw_folder_path)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]

    edited_folder = os.path.join(raw_folder_path, "tracking_results")
    os.makedirs(edited_folder, exist_ok=True)

    if not video_files:
        print("資料夾內沒有影片檔案")
        return

    print(f"共找到 {len(video_files)} 個影片檔，開始平行轉檔...")

    # 預設使用 CPU 核心數的一半，避免把系統壓到爆
    if max_workers is None:
        try:
            cpu_cnt = os.cpu_count() or 4
        except Exception:
            cpu_cnt = 4
        max_workers = max(1, cpu_cnt // 2)

    print(f"使用 {max_workers} 個平行進程")

    # 多進程平行處理
    errors = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                _convert_single_video,
                raw_folder_path,
                filename,
                TO_HEIGHT,
                TO_FPS,
                edited_folder
            ): filename
            for filename in video_files
        }

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                _ = future.result()
            except Exception as e:
                print(f"❌ 轉檔失敗：{filename}，錯誤：{e}")
                errors.append((filename, e))

    print("\n=== 轉檔結束 ===")
    if errors:
        print("下列檔案轉檔失敗：")
        for filename, e in errors:
            print(f" - {filename}: {e}")
    else:
        print("所有影片皆已成功轉換完成！")

# Tracking via YOLOv11m x BoT-SORT
########################
def tracking(args):
    """
    args: (video_source, CameraNo, out_dir)
    video_source: 不含副檔名的路徑或檔名 (你原本給 tracking 的那個)
    CameraNo: 用來命名輸出檔案
    """
    video_source, CameraNo, out_dir = args

    csv_path = out_dir / f"tracks_C{CameraNo}.csv"

    # --- open input to get FPS & size (for proper VideoWriter) ---
    input_path = os.path.join(out_dir, str(video_source) + IN_VIDEO_EXTENSION)
    print(input_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # --- load YOLO model (每個 process 建立自己的 model) ---
    print(f"[C{CameraNo}] loading model on device={DEVICE} ...")
    model = YOLO(WEIGHTS_DET)

    # --- open CSV/TXT ---
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "track_id", "x1", "y1", "x2", "y2",
        "width", "height", "conf", "class_id", "class_name"
    ])

    frame_idx = 0
    print(f"[C{CameraNo}] start tracking on {input_path}")

    for r in model.track(
        source=input_path,
        stream=False,           # False: 會直接跑完; True: generator
        conf=CONF,
        iou=IOU,
        classes=CLASSES,
        device=DEVICE,
        verbose=False
    ):
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clses = r.boxes.cls.detach().cpu().numpy().astype(int)
            ids = r.boxes.id

            if ids is None:
                ids_np = np.full((len(xyxy),), -1, dtype=int)
            else:
                ids_np = ids.detach().cpu().numpy().astype(int)

            for (x1, y1, x2, y2), tid, c, cid in zip(xyxy, ids_np, confs, clses):
                w = x2 - x1
                h = y2 - y1
                cname = model.names.get(int(cid), str(int(cid))) if hasattr(model, "names") else str(int(cid))

                # CSV
                csv_writer.writerow([
                    frame_idx, int(tid), int(x1), int(y1), int(x2), int(y2),
                    int(w), int(h), float(c), int(cid), cname
                ])
        frame_idx += 1

    # cleanup
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"[C{CameraNo}] ✅ Saved:")
    print(f" - CSV:   {csv_path}")

def get_targer_id(input_csv):
    max_frame = input_csv.frame.max()
    valid_id = input_csv.groupby("track_id").count()["frame"] > max_frame/4
    valid_id = valid_id.index[valid_id == True]
    
    id_group = input_csv[input_csv["track_id"].isin(valid_id)].groupby("track_id").aggregate([lambda x: x.iloc[0], lambda x: x.iloc[-1]])["x1"]
    id_group.columns = ["first_x", "last_x"]
    id_group["x_shift"] = id_group["last_x"] - id_group["first_x"]
    id_group = id_group["x_shift"] == id_group["x_shift"].max()
    target_id = id_group.index[id_group == True]
    out = input_csv[input_csv["track_id"] == target_id[0]]
    print(f"Target ID is {target_id[0]}")
    return out

def mack_up_value(in_df):
    empty = pd.DataFrame(data={"frame": range(in_df["frame"].min(), in_df["frame"].max()+1)})
    new = pd.merge(empty, in_df, how="left", on="frame")
    cols = ["track_id", "x1", "y1", "x2", "y2", "width", "height", 'camera', 'det_x1', 'det_x2', 'det_y1', 'det_y2']
    # print(new.dtypes)
    new[cols] = new[cols].infer_objects(copy=False)
    new[cols] = new[cols].interpolate()
    new.loc[new["class_name"].isna(), ["width", "height", "conf", "class_id"]] = None, None, None, None
    return new


def draw_pose_on_image(img, kps_xy_abs, boxes_xyxy_abs=None, kp_conf=None):
    """
    在原圖 img 上畫骨架與 bbox。
    kps_xy_abs: (n, 17, 2)  已加上位移後的 "原圖座標"
    boxes_xyxy_abs: (n, 4)  已加上位移後的 "原圖座標"
    kp_conf: (n, 17) or None
    """
    # 畫骨架
    
    # 常見骨架邊（可依需要調整）
    SKELETON_EDGES = [
        (5, 6),   # L-shoulder to R-shoulder
        (5, 7), (7, 9),   # L-shoulder -> L-elbow -> L-wrist
        (6, 8), (8,10),   # R-shoulder -> R-elbow -> R-wrist
        (11,12),          # L-hip to R-hip
        (5,11), (6,12),   # shoulders to hips
        (11,13), (13,15), # L-hip -> L-knee -> L-ankle
        (12,14), (14,16), # R-hip -> R-knee -> R-ankle
        (0,5), (0,6)      # nose to shoulders
    ]

    for p in range(kps_xy_abs.shape[0]):  # 每個人
        # 先畫線
        for a, b in SKELETON_EDGES:
            xa, ya = kps_xy_abs[p, a]
            xb, yb = kps_xy_abs[p, b]
            if np.isfinite([xa, ya, xb, yb]).all():
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

        # 再畫點
        for j in range(kps_xy_abs.shape[1]):
            x, y = kps_xy_abs[p, j]
            if np.isfinite([x, y]).all():
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

    # 畫 bbox
    if boxes_xyxy_abs is not None:
        for (x1, y1, x2, y2) in boxes_xyxy_abs:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

def append_rows(frame_idx, kps_xy_abs, boxes_xyxy_abs=None, det_conf=None, kp_conf=None):
    """
    將已加位移的原圖座標寫進 rows（之後匯出 CSV）
    """  
    n = kps_xy_abs.shape[0]
    for det_id in range(n):
        for j in range(kps_xy_abs.shape[1]):
            x, y = kps_xy_abs[det_id, j]
            kc = float(kp_conf[det_id, j]) if kp_conf is not None else np.nan
            if boxes_xyxy_abs is not None and det_id < len(boxes_xyxy_abs):
                x1, y1, x2, y2 = boxes_xyxy_abs[det_id]
            else:
                x1 = y1 = x2 = y2 = np.nan
            dconf = float(det_conf[det_id]) if det_conf is not None and det_id < len(det_conf) else np.nan
            rows.append({
                "frame": frame_idx,
                "det_id": det_id,
                "kp_index": j,
                "kp_name": KP_NAMES[j],
                "x": float(x), "y": float(y), "kp_conf": kc,
                "box_x1": float(x1), "box_y1": float(y1),
                "box_x2": float(x2), "box_y2": float(y2),
                "det_conf": dconf
            })

def pose_estimation_worker(args):
    """
    多進程 worker 版本的 pose estimation    
    args: (
        infile,               # 不含副檔名的影片路徑 (或檔名)
        camera_id,            # 相機編號 (原本的 camera_id)
        out_dir,              # Path 物件
        track_info_cam        # 已經過濾好、只剩這個 camera 的 track_info DataFrame
    )
    """
    (infile, camera_id, out_dir, track_info_cam) = args

    global rows  # append_rows 會用到這個名字

    output_file_name = f"{infile}_pose"
    # 只推論這些幀（以 0 為起點的 frame index），已經只含這個 camera
    frames_to_run = sorted(set(track_info_cam.frame.to_list()))

    # ======================= 讀取影片屬性 =======================
    cap = cv2.VideoCapture(infile + IN_VIDEO_EXTENSION)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {infile}{IN_VIDEO_EXTENSION}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 限制幀號在合法範圍
    frames_to_run = [f for f in frames_to_run if 0 <= f < total_frames]
    if not frames_to_run:
        raise ValueError(f"[camera {camera_id}] frames_to_run 內沒有合法的幀號。")

    # ======================= 準備輸出檔 =======================
    # 骨架座標輸出 CSV
    csv_path = Path(out_dir) / f"{output_file_name}_keypoints.csv"
    
    if os.path.exists(csv_path):
        print(f'File is exists: {csv_path}')

    else:
        rows = []  # 先放在記憶體，最後一次寫出

        # # 骨架座標輸出 CSV
        # csv_path = Path(out_dir) / f"{output_file_name}_keypoints.csv"
        # ======================= 載入 YOLO pose 模型 =======================
        print(f"[Pose C{camera_id}] loading pose model on {DEVICE} ...")
        model = YOLO(WEIGHTS_POSE)

        print(f"[Pose C{camera_id}] start pose estimation on {infile}{IN_VIDEO_EXTENSION}")

        # ========== 推論與繪製 ==========
        # 只輸出指定幀的影片（中間 frame 不寫）
        for fidx in frames_to_run:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                print(f"[Pose C{camera_id}] [Warn] cannot read frame {fidx}")
                continue

            # 1) 讀取 bbox（這裡 track_info_cam 只包含這個 camera）
            ti = track_info_cam[track_info_cam.frame == fidx]
            if ti.empty:
                print(f"[Pose C{camera_id}] [Warn] no det box at frame {fidx}")
                continue

            det_x1 = int(ti.det_x1.iloc[0])
            det_x2 = int(ti.det_x2.iloc[0])
            det_y1 = int(ti.det_y1.iloc[0])
            det_y2 = int(ti.det_y2.iloc[0])

            crop = frame[det_y1:det_y2, det_x1:det_x2].copy()

            # 2) 在裁切圖上做 pose
            res = model.predict(
                crop,
                conf=CONF,
                iou=IOU,
                device=DEVICE,      # ✅ GPU/CPU 控制在這裡
                verbose=False
            )[0]

            # 3) 取出關鍵點與框（裁切座標），並轉回原圖座標（+shift）
            kps_xy_abs = np.zeros((0, 17, 2), dtype=float)
            kps_conf = None

            if getattr(res, "keypoints", None) is not None and res.keypoints is not None:
                kps_xy = res.keypoints.xy.detach().cpu().numpy()  # (n,17,2)
                kps_xy_abs = kps_xy.copy()
                kps_xy_abs[..., 0] += det_x1
                kps_xy_abs[..., 1] += det_y1

                if getattr(res.keypoints, "conf", None) is not None:
                    kps_conf = res.keypoints.conf.detach().cpu().numpy()

            boxes_xyxy_abs = None
            det_conf = None
            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
                boxes_xyxy_abs = boxes_xyxy.copy()
                boxes_xyxy_abs[:, [0, 2]] += det_x1
                boxes_xyxy_abs[:, [1, 3]] += det_y1
                det_conf = res.boxes.conf.detach().cpu().numpy()

            # 4) 寫入影片與 CSV
            if kps_xy_abs.shape[0] > 0:
                append_rows(fidx, kps_xy_abs, boxes_xyxy_abs, det_conf, kps_conf)

        # ========= 收尾 =========
        cap.release()

        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"[Pose C{camera_id}] ✅ Done.")
        print(f"  Keypoints CSV saved to: {csv_path}")

def get_keypoint(camera_info, in_file):
    # Keep data within LR edge.
    # 暫時以軀幹(肩、髖)最靠前者(x)作為跑者當前位置；暫定五機拍攝
    if camera_info[0] == 1:
        turso_front = in_file.loc[(in_file.kp_index.isin([5, 6, 11, 12])) & (in_file.det_id == 0), ["frame", "x"]].groupby("frame").max() # in_file.det_id == 0 不一定，但通常會是，因為是在人的bounging box上找骨架；除非兩個人站一起
        turso_front = turso_front.reset_index()
        turso_front["distance"] = (turso_front["x"] - camera_info[1])*camera_info[4]
        turso_front["time"] = turso_front["frame"]*camera_info[6]
        turso_front = turso_front[(turso_front["distance"] <= 2000) & ((turso_front["distance"] >= 0))]
        turso_front["camera"] = camera_info[0]
        turso_front["time"] = turso_front["time"] - turso_front["time"].min()  # 時間重新對齊(令第一幀0)
        
        ankle_data = in_file.loc[(in_file.kp_index.isin([15, 16])) & (in_file.det_id == 0), ["frame", "kp_index", "kp_name", "x", "y", "box_x1", "box_x2"]]  # in_file.det_id == 0 不一定，但通常會是，因為是在人的bounging box上找骨架；除非兩個人站一起
        ankle_data = ankle_data.reset_index()
        ankle_data["distance"] = (ankle_data["x"] - camera_info[1])*camera_info[4]
        ankle_data["time"] = ankle_data["frame"]*camera_info[6]
        ankle_data = ankle_data[(ankle_data["distance"] <= 2000) & ((ankle_data["distance"] >= 0))]
        ankle_data["camera"] = camera_info[0]
        ankle_data["time"] = ankle_data["time"] - ankle_data["time"].min()  # 時間重新對齊(令第一幀0)
        ankle_data["box_width"] = ankle_data["box_x2"] - ankle_data["box_x1"]  # bounding box width
        # ankle_data = ankle_data.loc[ankle_data.groupby("frame")["y"].idxmax()]  # Method I, 保留max x for each frame
        ankle_data_sorted = ankle_data.sort_values(["frame", "x"], ascending=[True, False])  # Method II, 保留max x for each frame
        ankle_data = ankle_data_sorted.drop_duplicates(subset="frame", keep="first")

        # make up ankle data
        empty = pd.DataFrame(data={"frame": range(ankle_data["frame"].min(), ankle_data["frame"].max()+1)})
        ankle_data['camera'] = ankle_data['camera'].astype(int)
        new = pd.merge(empty, ankle_data, how="left", on="frame")
        ankle_data = new.interpolate()
        ankle_data.loc[ankle_data["kp_name"].isna(), ["kp_index", "kp_name"]] = None, None
        
    else:
        turso_front = in_file.loc[(in_file.kp_index.isin([5, 6, 11, 12])) & (in_file.det_id == 0), ["frame", "x"]].groupby("frame").max()
        turso_front = turso_front.reset_index()
        turso_front["distance"] = (int(camera_info[0])-1)*2000 + (turso_front["x"] - camera_info[1])*camera_info[4]
        turso_front["time"] = turso_front["frame"]*camera_info[6]
        turso_front = turso_front[(turso_front["distance"] <= (int(camera_info[0])*2000)) & (turso_front["distance"] > (int(camera_info[0])*2000-2000))]
        turso_front["camera"] = camera_info[0]
        turso_front["time"] = turso_front["time"] - turso_front["time"].min()  # 時間重新對齊(令第一幀0)
        
        ankle_data = in_file.loc[(in_file.kp_index.isin([15, 16])) & (in_file.det_id == 0), ["frame", "kp_index", "kp_name", "x", "y", "box_x1", "box_x2"]]  # in_file.det_id == 0 不一定，但通常會是，因為是在人的bounging box上找骨架；除非兩個人站一起
        ankle_data = ankle_data.reset_index()
        ankle_data["distance"] = (int(camera_info[0])-1)*2000 + (ankle_data["x"] - camera_info[1])*camera_info[4]
        ankle_data["time"] = ankle_data["frame"]*camera_info[6]
        ankle_data = ankle_data[(ankle_data["distance"] <= (int(camera_info[0])*2000)) & (ankle_data["distance"] > (int(camera_info[0])*2000-2000))]
        ankle_data["camera"] = camera_info[0]
        ankle_data["time"] = ankle_data["time"] - ankle_data["time"].min()  # 時間重新對齊(令第一幀0)
        ankle_data["box_width"] = ankle_data["box_x2"] - ankle_data["box_x1"]  # bounding box width
        # ankle_data = ankle_data.loc[ankle_data.groupby("frame")["y"].idxmax()]  # Method I, 保留max x for each frame
        ankle_data_sorted = ankle_data.sort_values(["frame", "x"], ascending=[True, False])  # Method II, 保留max x for each frame
        ankle_data = ankle_data_sorted.drop_duplicates(subset="frame", keep="first")
        
        # make up ankle data
        empty = pd.DataFrame(data={"frame": range(ankle_data["frame"].min(), ankle_data["frame"].max()+1)})
        ankle_data['camera'] = ankle_data['camera'].astype(int)
        new = pd.merge(empty, ankle_data, how="left", on="frame")
        ankle_data = new.interpolate()
        ankle_data.loc[ankle_data["kp_name"].isna(), ["kp_index", "kp_name"]] = None, None

    return turso_front, ankle_data


def step_frame(infile):
    # 假設你的 DataFrame 是 df，欄位 y 是原始訊號
    # print(infile)
    y = infile.groupby("frame").box_width.mean().reset_index().box_width.values
    x = infile.groupby("frame").box_width.mean().reset_index().frame.values
    # (可選) 先用 rolling median 去除尖刺
    y_med = (
        pd.Series(y)
        .rolling(window=3, center=True)
        .median()
        .bfill().ffill()
        .values
    )
    
    # 再用 Savitzky-Golay 做平滑 (類似低通濾波)
    # window_length 要是奇數，rough 建議 ≈ (總點數 / 預期波峰數) 的一半～1倍
    N = len(y_med)
    # print(N)
    window_length = 7  # |1 讓它變奇數
    polyorder = 2
    
    y_smooth = savgol_filter(y_med, window_length=window_length, polyorder=polyorder)
    
    # plt.plot(x, y_smooth,'ro-', linewidth=1, markersize=2) 
    # plt.show()
    
    # distance 設一個「最小間隔」，避免太密的小抖動
    peaks_all_temp, props_all = find_peaks(
        y_smooth,
        distance=10,            # 粗略，例如總長度/100；可以再調
        prominence=None               # 先不設門檻，把 prominence 全抓出來
    )

    # Bounding Box Width Peak所在位置    
    peaks_all_temp = peaks_all_temp + infile.frame.min()

    # print(peaks_all)
    return peaks_all_temp



#########################
# Part III: Generate Final Video (Define function)
#########################
def track_and_draw(raw_folder_path: str, out_final_name: str, video_count: int, progress_callback=None):
    # 資料夾路徑
    tracking_result_folder_path = os.path.join(raw_folder_path, "tracking_results")
    video_source_list = [f"cam{i}" for i in range(1, video_count + 1)]
    CameraNo_list = list(range(1, video_count + 1))
    out_dir = Path(tracking_result_folder_path)   # where to save files
    out_dir.mkdir(parents=True, exist_ok=True)

    # 20251107
    all_camera_info = [
        lambda: get_edge_fps(os.path.join(raw_folder_path, f"cam1{IN_VIDEO_EXTENSION}"), 1, 450, 3440, 1955),
        lambda: get_edge_fps(os.path.join(raw_folder_path, f"cam2{IN_VIDEO_EXTENSION}"), 2, 375, 3440, 1890),
        lambda: get_edge_fps(os.path.join(raw_folder_path, f"cam3{IN_VIDEO_EXTENSION}"), 3, 105, 1145, 630),
        lambda: get_edge_fps(os.path.join(raw_folder_path, f"cam4{IN_VIDEO_EXTENSION}"), 4, 190, 1730, 955),
        lambda: get_edge_fps(os.path.join(raw_folder_path, f"cam5{IN_VIDEO_EXTENSION}"), 5, 140, 1785, 960),
    ]

    camera_info = [all_camera_info[i]() for i in range(video_count)]
    if progress_callback: progress_callback(5)
    reformatting_videos(raw_folder_path)
    if progress_callback: progress_callback(15)

    ############
    ### Multiple Processing - Tracking
    ############
    tasks = [
        (video_source, CameraNo, out_dir)
        for video_source, CameraNo in zip(video_source_list, CameraNo_list)
    ]
    print(f"Start multiprocessing tracking with {MAX_WORKERS_DET} workers on device={DEVICE}")

    errors = []
    t0 = time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS_DET) as executor:
        futures = [executor.submit(tracking, args) for args in tasks]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print("❌ worker error:", e)
                errors.append(e)

    dt = time() - t0
    print(f"\n=== All done in {dt:.1f} sec ===")
    if errors:
        print(f"{len(errors)} task(s) failed.")
    else:
        print("All videos processed successfully.")
    if progress_callback: progress_callback(40)

    
    ############
    ### Deal with tracked information. (Find out 1.5x bounding box)
    ############
    split_tracker = []
    for cid in CameraNo_list:
        df = pd.read_csv(f"{tracking_result_folder_path}/tracks_C{cid}.csv")
        df = get_targer_id(df)
        df["camera"] = cid
        split_tracker.append(df)
    full_tracker = pd.concat(split_tracker)

    del cid, split_tracker

    # Calculate detect edge of 1.5x bounding box
    full_tracker["det_x1"] = full_tracker["x1"] - full_tracker["width"]/4
    full_tracker["det_x2"] = full_tracker["x2"] + full_tracker["width"]/4
    full_tracker["det_y1"] = full_tracker["y1"] - full_tracker["height"]/4
    full_tracker["det_y2"] = full_tracker["y2"] + full_tracker["height"]/4

    full_tracker["det_x1"] = full_tracker["det_x1"].astype(int)
    full_tracker["det_x2"] = full_tracker["det_x2"].astype(int)
    full_tracker["det_y1"] = full_tracker["det_y1"].astype(int)
    full_tracker["det_y2"] = full_tracker["det_y2"].astype(int)

    full_tracker.loc[full_tracker["det_x1"] < 0, "det_x1"] = 0
    full_tracker.loc[full_tracker["det_y1"] < 0, "det_x2"] = 0
    full_tracker.loc[full_tracker["det_x2"] < 0, "det_y1"] = 0
    full_tracker.loc[full_tracker["det_y2"] < 0, "det_y2"] = 0

    track_list = []
    for cid in CameraNo_list:
        track_list.append(mack_up_value(full_tracker[full_tracker.camera == cid]))
    track_info = pd.concat(track_list)
    del track_list, cid

    
    ############
    ### Multiple Processing - Pose
    ############
    os.chdir(tracking_result_folder_path)

    args_list = []
    for infile, cid in zip(video_source_list, CameraNo_list):
        # 只切出這個 camera 的 track_info，減少跨 process 傳輸量
        track_info_cam = track_info[track_info.camera == cid].copy()
        args = (
            infile,
            cid,
            out_dir,
            track_info_cam
        )
        args_list.append(args)

    print(f"Start pose multiprocessing with {MAX_WORKERS_POSE} workers on device={DEVICE}")

    errors = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_POSE) as executor:
        futures = [executor.submit(pose_estimation_worker, a) for a in args_list]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print("❌ worker error:", e)
                errors.append(e)

    print("\n=== Pose estimation all done ===")
    if errors:
        print(f"{len(errors)} task(s) failed.")
    else:
        print("All pose jobs completed successfully.")
    if progress_callback: progress_callback(80)


    ############
    ### Combine the pose-tracker info. (turso/full & ankle/for steps)
    ############
    full_tracker = []
    ankle_tracker = []
    for infile, cid in zip(video_source_list, CameraNo_list):
        turso, ankle = get_keypoint(camera_info[int(cid)-1], pd.read_csv(tracking_result_folder_path + "/" + infile + "_pose_keypoints.csv"))
        full_tracker.append(turso)
        ankle_tracker.append(ankle)
        del turso, ankle

    for cid in CameraNo_list:
        if cid == CameraNo_list[0]:
            tracked100m = full_tracker[int(cid)-1]     
            ankle100m = ankle_tracker[int(cid)-1]   
        else:
            full_tracker[int(cid)-1]["time"] = full_tracker[int(cid)-1]["time"] + tracked100m["time"].max()
            tracked100m = pd.concat([tracked100m, full_tracker[int(cid)-1][1:]])
            ankle_tracker[int(cid)-1]["time"] = ankle_tracker[int(cid)-1]["time"] + tracked100m["time"].max()
            ankle100m = pd.concat([ankle100m, ankle_tracker[int(cid)-1]])

    tracked100m = tracked100m.reset_index(drop=True)
    ankle100m = ankle100m.reset_index(drop=True)
    

    # Extract Step's frame (via bounding box width peak)
    step_full_temp = []
    for cid in CameraNo_list:
        step_full_temp.append(pd.DataFrame(data={"camera": cid, "step_frame": step_frame(ankle100m[ankle100m.camera == cid])}))
    step_full_temp = pd.concat(step_full_temp).reset_index(drop=True)

    # Bounding Box Width Peak往後十幀內的y max (先試不分腳) 很可能會接近腳的落點
    step_full = []
    for i in range(0, len(step_full_temp)):
        temp_f = step_full_temp.iloc[i].step_frame
        temp_cid = step_full_temp.iloc[i].camera
        temp_ankle = ankle100m.loc[(ankle100m.camera == temp_cid) & (ankle100m.frame.isin(range(temp_f, temp_f+11)))]
        temp_ankle_sorted = temp_ankle.sort_values(["y"], ascending=[False]).reset_index(drop=True)[["camera", "frame"]]  # 保留max y
        temp_ankle_sorted.columns = ["camera", "step_frame"]
        step_full.append(temp_ankle_sorted[0:1])
        print(f"original cid & frame: {temp_cid}, {temp_f}; new: {temp_ankle_sorted.camera[0]}, {temp_ankle_sorted.step_frame[0]}")
        del temp_f, temp_cid, temp_ankle, temp_ankle_sorted
    step_full = pd.concat(step_full)
    print(step_full)
    

    # Velocity
    tracked100m = tracked100m.sort_values('time').reset_index(drop=True)
    time_new = np.linspace(tracked100m['time'].min(), tracked100m['time'].max(), 30)
    dist_interp = np.interp(time_new, tracked100m['time'], tracked100m['distance'])
    velocity = np.diff(dist_interp) / np.diff(time_new)
    vel_df = pd.DataFrame({
        'time': time_new[1:],        # midpoints (since diff shortens by 1)
        'velocity_m_per_s': velocity/100
    })
    
    v_smooth = savgol_filter(vel_df.velocity_m_per_s, window_length=11, polyorder=2)


    velocity_df = pd.DataFrame(data={'time': vel_df.time, 'velocity_m_per_s': v_smooth})
    time_new = np.linspace(0, velocity_df['time'].max(), len(tracked100m))
    vel_interp = np.interp(time_new, velocity_df['time'], velocity_df['velocity_m_per_s'])


    acc = np.diff(vel_interp) / np.diff(time_new)

    tracked100m["velocity"] = vel_interp
    tracked100m["acc"] = np.insert(acc, 0, 0)

    tracked100m = tracked100m.loc[(tracked100m.time >= tracked100m[tracked100m.distance > 0].head(1).time.values[0]) & (tracked100m.time >= tracked100m[tracked100m.velocity > 0].head(1).time.values[0])]
    tracked100m["new_time"] = tracked100m["time"] - tracked100m["time"].min()

    a_smooth = savgol_filter(tracked100m.acc, window_length=15, polyorder=2)

    # Savitzky-Golay 平滑
    # 重新取樣後再繪製(也可以用moving window / 濾波方式)
    n_t_o = np.linspace(0, tracked100m['new_time'].max(), 50)
    acc_interp = np.interp(n_t_o, tracked100m['new_time'], tracked100m['acc'])
    n_t = np.linspace(0, tracked100m['new_time'].max(), len(np.unique(tracked100m.time, return_index=False)))
    acc_interp = np.interp(n_t, n_t_o, acc_interp)


    a_smooth = savgol_filter(acc_interp, window_length=11, polyorder=2)

    tracked100m["acc_smooth"] = a_smooth

    tracked100m.to_csv(f"track_pose_full_data.csv", sep=",", index=False)
    step_full.to_csv(f"step_full_data.csv", sep=",", index=False)
    ankle100m.to_csv(f"track_ankle_full_data.csv", sep=",", index=False)
    if progress_callback: progress_callback(90)


    ############
    ### Create Final Video (Combine video / step point / step distance / Timer & Plots)
    ############

    # ========== 1. 讀取 CSV ==========
    os.chdir(tracking_result_folder_path)
    tracked100m = pd.read_csv(f"track_pose_full_data.csv")
    step_full = pd.read_csv(f"step_full_data.csv")
    ankle100m = pd.read_csv(f"track_ankle_full_data.csv")
    tracked100m['distance'] = tracked100m['distance']/100
    ankle100m['distance'] = ankle100m['distance']/100
    # 假設欄位為：video_id, frame_idx, time, distance, velocity, accel
    # 依時間排序（確保影片輸出順序跟時間一致）
    tracked100m = tracked100m.sort_values("new_time").reset_index(drop=True)

    # 方便後面用 numpy
    times_all = tracked100m["new_time"].values
    dist_all  = tracked100m["distance"].values
    vel_all   = tracked100m["velocity"].values
    acc_all   = tracked100m["acc_smooth"].values

    # ========== 2. 開啟所有影片 ==========
    video_ids = sorted(tracked100m["camera"].unique())
    caps = {}
    for vid in video_ids:
        cap = cv2.VideoCapture(VIDEO_PATTERN.format(int(vid)))
        if not cap.isOpened():
            raise RuntimeError(f"Video {VIDEO_PATTERN.format(int(vid))} cannot be opened")
        caps[int(vid)] = cap

    # 用其中一支影片取得影像尺寸 & fps（假設它們都一樣）
    sample_cap = caps[int(video_ids[0])]
    frame_width  = int(sample_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(sample_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = sample_cap.get(cv2.CAP_PROP_FPS) or TO_FPS  # 若讀不到就用 60 FPS

    # ========== 3. 準備 Matplotlib 圖（右半邊） ==========
    plt.ioff()  # 關閉互動模式，不彈視窗
    fig, axes = plt.subplots(3, 1, figsize=(5, 8))  # 右邊圖的比例

    canvas = FigureCanvas(fig)

    ylabels = ["Distance (m)", "Velocity (m/s)", "Acceleration (m/s^2)"]

    for ax, yl in zip(axes, ylabels):
        ax.set_ylabel(yl, fontsize=LABEL_FONTSIZE, fontname=FONT_NAME)
        ax.tick_params(labelsize=TICK_FONTSIZE)

    axes[-1].set_xlabel("Time (s)", fontsize=LABEL_FONTSIZE, fontname=FONT_NAME)


    # 設定固定 y 範圍（可依你資料調整或註解掉再 auto scale）
    # axes[0].set_ylim(dist_all.min(), dist_all.max())
    # axes[1].set_ylim(vel_all.min(),  vel_all.max())
    # axes[2].set_ylim(acc_all.min(),  acc_all.max())
    t_min, t_max = times_all.min()-(times_all.max()-times_all.min())*0.05, times_all.max()+(times_all.max()-times_all.min())*0.05
    d_min, d_max = dist_all.min()-(dist_all.max()-dist_all.min())*0.05, dist_all.max()+(dist_all.max()-dist_all.min())*0.05
    v_min, v_max = vel_all.min()-(vel_all.max()-vel_all.min())*0.05, vel_all.max()+(vel_all.max()-vel_all.min())*0.05
    a_min, a_max = acc_all.min()-(acc_all.max()-acc_all.min())*0.05, acc_all.max()+(acc_all.max()-acc_all.min())*0.05

    for ax in axes:
        ax.set_xlim(t_min, t_max)
        
    # ========== 4. 建立輸出影片 ==========
    out_width  = frame_width
    out_height = frame_height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_final_name + ".mp4", fourcc, fps_out, (out_width, out_height))

    # ========== 5. 主迴圈：逐列輸出畫面 ==========
    for i, row in tracked100m.iterrows():
        vid_id     = int(row["camera"])
        frame_idx  = int(row["frame"])
        curr_time  = row["new_time"]

        cap = caps[vid_id]
        # 跳到該幀
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[Warning] Cannot read frame {frame_idx} from video {vid_id}")
            continue

    # 步伐點位標記    
        if frame_idx >= step_full[step_full["camera"] == vid_id].step_frame.values.min():
            current_frame_list = step_full[(step_full["camera"] == vid_id) & (step_full["step_frame"] <= frame_idx)].step_frame.values
            step = 1
            stepmax = len(current_frame_list)
            for fid in current_frame_list:
                if step == 1:
                    try:
                        current_dis = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].distance.values[0]
                        current_x = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].x.values[0]
                        current_y = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].y.values[0]
                        cv2.circle(frame, (int(current_x), int(current_y)+10), 7, (255, 255, 0), -1)
                        step += 1
                        if step > stepmax:
                            break
                    except:
                        print(f'cannot find fid:{fid} in camera:{vid_id} ankle.csv') 
                        pass
                elif step%2 == 0:
                    try:
                        dis_diff = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].distance.values[0] - current_dis
                        x_diff = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].x.values[0] - current_x
                        current_x = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].x.values[0]
                        current_y = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].y.values[0]
                        cv2.circle(frame, (int(current_x), int(current_y)+10), 7, (255, 255, 0), -1)
                        cv2.putText(
                            frame,
                            f"{dis_diff:.2f} m", # 文字
                            (int(current_x) - int(x_diff), int(ankle100m[(ankle100m["camera"] == vid_id)].y.max()) + 40),     # 位置（可微調）
                            cv2.FONT_HERSHEY_SIMPLEX,   # 字體
                            0.8,                        # 字體大小
                            (0, 255, 255),              # 顏色 (BGR) = 黃色
                            2,                          # 線條粗細
                            cv2.LINE_AA
                        )
                        
                        step += 1
                        current_dis = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].distance.values[0]
                        if step > stepmax:
                            break                
                    except:
                        print(f'cannot find fid:{fid} in camera:{vid_id} ankle.csv') 
                        pass
                else:
                    try:                    
                        dis_diff = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].distance.values[0] - current_dis
                        x_diff = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].x.values[0] - current_x
                        current_x = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].x.values[0]
                        current_y = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].y.values[0]
                        cv2.circle(frame, (int(current_x), int(current_y)+10), 7, (255, 255, 0), -1)
                        cv2.putText(
                            frame,
                            f"{dis_diff:.2f} m", # 文字
                            (int(current_x) - int(x_diff), int(ankle100m[(ankle100m["camera"] == vid_id)].y.max()) + 80),     # 位置（可微調）
                            cv2.FONT_HERSHEY_SIMPLEX,   # 字體
                            0.8,                        # 字體大小
                            (0, 255, 255),              # 顏色 (BGR) = 黃色
                            2,                          # 線條粗細
                            cv2.LINE_AA
                        )
                        
                        step += 1
                        current_dis = ankle100m[(ankle100m["camera"] == vid_id) & (ankle100m["frame"] == fid)].distance.values[0]
                        if step > stepmax:
                            break      
                    except:
                        print(f'cannot find fid:{fid} in camera:{vid_id} ankle.csv') 
                        pass

        # 設定右上角計時器文字
        timer_text = f"Timer: {curr_time:.2f} s"
        timer_shift_text = f"(Shift: -{tracked100m[tracked100m['velocity']>0].new_time.values[0]:.2f} s)"
        
        # (x, y) = 右上角靠左 180px、向下 40px
        cv2.putText(
            frame,
            timer_text,
            (frame_width - 350, 40),     # 位置（可微調）
            cv2.FONT_HERSHEY_SIMPLEX,   # 字體
            1.2,                        # 字體大小
            (0, 255, 255),              # 顏色 (BGR) = 黃色
            3,                          # 線條粗細
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            timer_shift_text,
            (frame_width - 350, 80),     # 位置（可微調）
            cv2.FONT_HERSHEY_SIMPLEX,   # 字體
            0.8,                        # 字體大小
            (0, 255, 255),              # 顏色 (BGR) = 黃色
            2,                          # 線條粗細
            cv2.LINE_AA
        )

        # ====== 5-5. 左右拼接，寫入輸出影片 ======
        out.write(frame)

    # ========== 6. 資源釋放 ==========
    for cap in caps.values():
        cap.release()
    out.release()
    plt.close(fig)

    print(f"✅ Done! 輸出為 {out_final_name}.mp4") 

    # === 7. 計算 gait statistics ===
    gait_stats = compute_gait_statistics(
        os.path.join(tracking_result_folder_path, "track_pose_full_data.csv"),
        os.path.join(tracking_result_folder_path, "step_full_data.csv"),
    )

    cmd = [
        "ffmpeg", "-i", out_final_name + ".mp4",
        "-c:v", "libx264",
        "-c:a", "aac",
        out_final_name + "_meta.mp4"
    ]
    subprocess.run(cmd, check=True)

    return gait_stats

def compute_gait_statistics(
    track_csv_path: str,
    step_csv_path: str,
    camera_id: int | None = None
):
    """
    回傳：
    {
        total_time,
        avg_velocity,
        avg_acceleration,
        avg_step_length
    }
    """

    # === 讀取資料 ===
    track_df = pd.read_csv(track_csv_path)
    step_df = pd.read_csv(step_csv_path)

    # === 若有多 camera，先過濾 ===
    if camera_id is not None:
        track_df = track_df[track_df["camera"] == camera_id]
        step_df = step_df[step_df["camera"] == camera_id]

    track_df = track_df.sort_values("frame").reset_index(drop=True)

    # === 總時間 ===
    total_time = track_df["new_time"].iloc[-1] - track_df["new_time"].iloc[0]

    # === 平均速度 ===
    avg_velocity = track_df["velocity"].mean()

    # === 平均加速度（使用平滑後）===
    avg_acceleration = track_df["acc_smooth"].mean()

    # === 平均步幅 ===
    # 將 step frame 對應到 distance
    step_frames = step_df["step_frame"].astype(int).tolist()

    step_distances = track_df[
        track_df["frame"].isin(step_frames)
    ][["frame", "distance"]].sort_values("frame")

    # 計算相鄰步距
    step_lengths = step_distances["distance"].diff().dropna()

    avg_step_length = step_lengths.mean() if len(step_lengths) > 0 else 0.0

    return {
        "total_time": float(total_time),
        "avg_velocity": float(avg_velocity),
        "avg_acceleration": float(avg_acceleration),
        "avg_step_length": float(avg_step_length),
    }

if __name__ == "__main__":
    root_folder = "/home/hsuanya/workspace/running_analysis/backend/data/run_sessions/d9532b78-1a72-4101-8c9f-1ec812f8bd3b/b857ca3d-d7ed-42be-ab7a-4501203a441e"
    out_final_name = "analyzed_video"
    camera_count = 3
    track_and_draw(root_folder, out_final_name, camera_count)