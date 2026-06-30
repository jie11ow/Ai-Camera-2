#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, glob, time, json, hashlib, numpy as np, onnxruntime as ort, random, threading
from PIL import Image, ImageDraw, ImageFont

# ===================== 配置 =====================
MODEL_PATH   = '/home/abysm/camera/yolov8n.onnx'
SCENE_MODEL  = '/home/abysm/camera/places365_resnet18.onnx'
SCENE_MAP    = '/home/abysm/camera/places365_scene_mapping.json'
SAVE_DIR     = '/mnt/sdcard/photos/'
CAMERA_INDEX = 20
CONF_THRES   = 0.4
IOU_THRES    = 0.45
AI_INTERVAL_SEC = 0.6
MAX_CAM_INDEX   = 25

# GPIO 按钮映射
GPIO_BUTTONS = {
    70: 's',   # 拍照
    71: 'l',   # 相册
    72: 'g',   # 黄金螺旋
    73: 'q',   # 退出
    91: 'a',   # 上一张
    92: 'd',   # 下一张
}
BUTTON_ACTIVE_LOW = True

# ===================== 加载模型 =====================
session = ort.InferenceSession(MODEL_PATH)
inp = session.get_inputs()[0]
INPUT_NAME = inp.name
IMGSZ = inp.shape[2]
OUTPUT_NAMES = [o.name for o in session.get_outputs()]
print(f"✅ 检测模型 {IMGSZ}x{IMGSZ}")

scene_sess = ort.InferenceSession(SCENE_MODEL)
SCENE_IMGSZ = 224
print(f"✅ 场景模型 {SCENE_IMGSZ}x{SCENE_IMGSZ} (Places365)")

with open(SCENE_MAP) as f:
    scene_mapping = {int(k): v for k, v in json.load(f).items()}

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

CN_NAMES = {
    'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车',
    'airplane': '飞机', 'bus': '公交车', 'train': '火车', 'truck': '卡车',
    'boat': '船', 'traffic light': '红绿灯', 'fire hydrant': '消防栓',
    'stop sign': '停车标志', 'parking meter': '停车计时器', 'bench': '长椅',
    'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马', 'sheep': '羊',
    'cow': '牛', 'elephant': '大象', 'bear': '熊', 'zebra': '斑马',
    'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞', 'handbag': '手提包',
    'tie': '领带', 'suitcase': '行李箱', 'frisbee': '飞盘', 'skis': '滑雪板',
    'snowboard': '单板滑雪', 'sports ball': '球', 'kite': '风筝',
    'baseball bat': '棒球棒', 'baseball glove': '棒球手套', 'skateboard': '滑板',
    'surfboard': '冲浪板', 'tennis racket': '网球拍', 'bottle': '瓶子',
    'wine glass': '酒杯', 'cup': '杯子', 'fork': '叉子', 'knife': '刀',
    'spoon': '勺子', 'bowl': '碗', 'banana': '香蕉', 'apple': '苹果',
    'sandwich': '三明治', 'orange': '橙子', 'broccoli': '西兰花',
    'carrot': '胡萝卜', 'hot dog': '热狗', 'pizza': '披萨', 'donut': '甜甜圈',
    'cake': '蛋糕', 'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽',
    'bed': '床', 'dining table': '餐桌', 'toilet': '马桶', 'tv': '电视',
    'laptop': '笔记本电脑', 'mouse': '鼠标', 'remote': '遥控器',
    'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉',
    'oven': '烤箱', 'toaster': '烤面包机', 'sink': '水槽',
    'refrigerator': '冰箱', 'book': '书', 'clock': '时钟', 'vase': '花瓶',
    'scissors': '剪刀', 'teddy bear': '泰迪熊', 'hair drier': '吹风机',
    'toothbrush': '牙刷'
}

# ===================== 中文字体 =====================
def find_chinese_font():
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path): return path
    return None

# ===================== AI 推理 =====================
def preprocess(img):
    h0, w0 = img.shape[:2]
    r = IMGSZ / max(h0, w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    img = cv2.resize(img, (new_w, new_h))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = img
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    return blob, r, (h0, w0)

def nms(boxes, confidences):
    if len(boxes) == 0: return []
    x1 = boxes[:, 0] - boxes[:, 2]/2
    y1 = boxes[:, 1] - boxes[:, 3]/2
    x2 = boxes[:, 0] + boxes[:, 2]/2
    y2 = boxes[:, 1] + boxes[:, 3]/2
    areas = (x2 - x1) * (y2 - y1)
    order = confidences.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= IOU_THRES)[0]
        order = order[inds + 1]
    return keep

def detect(img):
    blob, r, (orig_h, orig_w) = preprocess(img)
    outputs = session.run(OUTPUT_NAMES, {INPUT_NAME: blob})
    output = outputs[0]
    predictions = np.squeeze(output).T
    boxes = predictions[:, :4]; scores = predictions[:, 4:]
    confidences = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    mask = confidences > CONF_THRES
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]
    indices = nms(boxes, confidences)
    results = []
    for i in indices:
        cx, cy, w_box, h_box = boxes[i]
        cx_orig, cy_orig = cx/r, cy/r
        w_orig, h_orig = w_box/r, h_box/r
        x1 = int(cx_orig - w_orig/2); y1 = int(cy_orig - h_orig/2)
        x2 = int(cx_orig + w_orig/2); y2 = int(cy_orig + h_orig/2)
        conf = confidences[i]; cls_id = class_ids[i]
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else 'unknown'
        results.append((x1, y1, x2, y2, conf, name))
    return results

def classify_scene(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 60:
        return "夜景"
    h, w = img.shape[:2]
    size = SCENE_IMGSZ
    if h > w:
        new_h = int(h * size / w); new_w = size
    else:
        new_h = size; new_w = int(w * size / h)
    img_ = cv2.resize(img, (new_w, new_h))
    startx = new_w//2 - size//2; starty = new_h//2 - size//2
    img_ = img_[starty:starty+size, startx:startx+size]
    blob = img_.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    out = scene_sess.run([scene_sess.get_outputs()[0].name],
                         {scene_sess.get_inputs()[0].name: blob})
    pred_idx = np.argmax(out[0][0])
    return scene_mapping.get(pred_idx, "普通")

# ===================== 构图规则 =====================
def detect_horizon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    if lines is None: return 0.0, False
    angles = []
    for rho, theta in lines[:,0]:
        a = np.cos(theta); b = np.sin(theta)
        x0 = a*rho; y0 = b*rho
        x1 = int(x0+1000*(-b)); y1 = int(y0+1000*a)
        x2 = int(x0-1000*(-b)); y2 = int(y0-1000*a)
        angle = np.arctan2(y2-y1, x2-x1)*180/np.pi
        if abs(angle) < 20: angles.append(angle)
    if len(angles) < 3: return 0.0, False
    return np.mean(angles), True

def check_headroom(frame, detections):
    persons = [d for d in detections if d[5]=='person']
    if not persons: return None
    best = max(persons, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
    h, _ = frame.shape[:2]
    headroom = best[1] / h
    if headroom < 0.05: return "tight"
    elif headroom > 0.3: return "loose"
    return None

def check_clutter(frame, detections):
    if not detections: return None
    h, w = frame.shape[:2]
    total_area = sum((d[2]-d[0])*(d[3]-d[1]) for d in detections)
    area_ratio = total_area / (h*w)
    count = len(detections)
    if count > 8: return "clutter"
    if area_ratio > 0.5: return "too_big"
    if count==1 and area_ratio<0.03: return "too_small"
    return None

def check_symmetry(frame):
    h, w = frame.shape[:2]
    if w<4: return None
    left = frame[:,:w//2]; right = cv2.flip(frame[:,w//2:], 1)
    diff = cv2.absdiff(left, right); score = 1 - (np.mean(diff)/255)
    if score > 0.7: return f"画面高度对称 (对称度 {score:.0%})，可采用对称构图"
    return None

def check_color_balance(frame):
    b,g,r = cv2.split(frame)
    mb,mg,mr = np.mean(b), np.mean(g), np.mean(r)
    if mb>mg*1.3 and mb>mr*1.3: return "blue"
    if mr>mg*1.3 and mr>mb*1.3: return "red"
    return None

def generate_hints(frame, detections, scene):
    h, w = frame.shape[:2]
    hints = []
    if scene != "普通":
        hints.append(f"当前场景：{scene}")
    if detections:
        best = max(detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
        cx = (best[0]+best[2])/2/w; cy = (best[1]+best[3])/2/h
        if abs(cx-0.5)<0.12 and abs(cy-0.5)<0.12:
            hints.append("主体居中，可尝试偏移至三分线")
        else:
            dh = '右' if cx < 0.3 else '左' if cx > 0.7 else ''
            dv = '下' if cy < 0.3 else '上' if cy > 0.7 else ''
            hints.append(f"主体偏{dh}{dv}，向三分线交点移动")
    else:
        hints.append("未检测到主体，请对准人物或景物")
    angle, ok = detect_horizon(frame)
    if ok and abs(angle) > 2.0:
        d = "顺时针" if angle > 0 else "逆时针"
        hints.append(f"地平线倾斜 {abs(angle):.1f}°，请{d}旋转")
    hm = check_headroom(frame, detections)
    if hm: hints.append("头顶空间" + ("过紧" if hm=="tight" else "过多"))
    sm = check_symmetry(frame)
    if sm: hints.append(sm)
    col = check_color_balance(frame)
    if col: hints.append("画面偏蓝" if col=="blue" else "画面偏红")
    return hints[:3]

# ===================== PIL 文字 =====================
def pil_puttext(img, text, position, color, font_size=22):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    font_path = find_chinese_font()
    if font_path:
        try: font = ImageFont.truetype(font_path, font_size)
        except: font = ImageFont.load_default()
    else: font = ImageFont.load_default()
    draw.text(position, text, fill=color+(255,), font=font)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)[:,:,:3]
    return np.ascontiguousarray(result)

# ===================== 黄金螺旋 =====================
def draw_golden_spiral(frame):
    h, w = frame.shape[:2]
    cx = int(w * 0.618); cy = int(h * 0.618)
    base = min(w, h) * 0.02
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    radii = [base * f for f in fib]
    pts = [(cx, cy)]
    angle = 0
    for r in radii:
        for t in np.linspace(0, np.pi/2, 20):
            px = cx + r * np.cos(angle + t); py = cy + r * np.sin(angle + t)
            pts.append((int(px), int(py)))
        angle += np.pi/2
        cx, cy = int(px), int(py)
    overlay = frame.copy()
    for i in range(1, len(pts)):
        cv2.line(overlay, pts[i-1], pts[i], (255, 255, 255), 2)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    cv2.circle(frame, (int(w*0.618), int(h*0.618)), 8, (0, 255, 255), -1)
    return frame

# ===================== 摄像头打开 =====================
def open_camera():
    cap = cv2.VideoCapture(f"/dev/video{CAMERA_INDEX}")
    if cap.isOpened():
        ret, _ = cap.read()
        if ret: return cap
        cap.release()
    for i in range(MAX_CAM_INDEX+1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: return cap
            cap.release()
    cap = cv2.VideoCapture("/dev/video0")
    if cap.isOpened():
        ret, _ = cap.read()
        if ret: return cap
        cap.release()
    return None

# ===================== GPIO 驱动 =====================
GPIO_LAST_STATE = {}

def init_gpio():
    for pin in GPIO_BUTTONS:
        gpio_path = f"/sys/class/gpio/gpio{pin}"
        if not os.path.exists(gpio_path):
            try:
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(pin))
            except PermissionError:
                print(f"⚠️ 无权限导出 GPIO {pin}，请以 root 运行或检查 udev 规则。")
                continue
            time.sleep(0.1)
        try:
            with open(f"/sys/class/gpio/gpio{pin}/direction", "w") as f:
                f.write("in")
        except:
            pass
        try:
            with open(f"/sys/class/gpio/gpio{pin}/value", "r") as f:
                GPIO_LAST_STATE[pin] = int(f.read().strip())
        except:
            GPIO_LAST_STATE[pin] = 1
    print("✅ GPIO 按钮初始化完成。")

def check_gpio_buttons():
    for pin, func in GPIO_BUTTONS.items():
        gpio_file = f"/sys/class/gpio/gpio{pin}/value"
        try:
            with open(gpio_file, "r") as f:
                val = int(f.read().strip())
        except:
            continue
        last = GPIO_LAST_STATE.get(pin, 1)
        trigger = (last == 1 and val == 0) if BUTTON_ACTIVE_LOW else (last == 0 and val == 1)
        GPIO_LAST_STATE[pin] = val
        if trigger:
            print(f"🔘 GPIO {pin} 按下，功能：{func}")
            return func
    return None

# ===================== 全局状态 =====================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detections = []
        self.scene = "普通"
        self.hints = ["AI 引擎启动中..."]
        self.latest_frame = None
        self.running = True

state = SharedState()

def ai_worker():
    while state.running:
        with state.lock:
            frame = state.latest_frame
            if frame is not None: frame = frame.copy()
        if frame is not None:
            try:
                dets = detect(frame)
                scene = classify_scene(frame)
                hints = generate_hints(frame, dets, scene)
                with state.lock:
                    state.detections = dets
                    state.scene = scene
                    state.hints = hints
            except Exception as e:
                print(f"AI 线程错误: {e}")
        time.sleep(AI_INTERVAL_SEC)

# ===================== 主程序 =====================
def main():
    init_gpio()
    cap = open_camera()
    if cap is None:
        print("❌ 未找到可用摄像头"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    ai_thread = threading.Thread(target=ai_worker, daemon=True)
    ai_thread.start()

    print("📷 AI 相机 (GPIO版) | 物理按钮已启用")
    cv2.namedWindow('AI Camera', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('AI Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0
    show_spiral = False
    view_mode = "camera"
    photo_list = []
    current_photo_idx = 0
    last_scene = "普通"
    last_hints = ["AI 引擎启动中..."]

    while True:
        if view_mode == "camera":
            ret, frame = cap.read()
            if not ret: continue
            raw_frame = frame.copy()
            display_frame = cv2.resize(frame, (640, 480))

            with state.lock:
                state.latest_frame = display_frame.copy()
                detections = state.detections
                scene = state.scene
                hints = state.hints

            if hints and hints[0] != "AI 引擎启动中...":
                last_scene = scene; last_hints = hints
            else:
                scene = last_scene; hints = last_hints

            if show_spiral:
                display_frame = draw_golden_spiral(display_frame)

            display_frame = pil_puttext(display_frame, f'场景: {scene}', (10,10), (220,220,220), 20)
            y_off = 35
            for hint in hints[:3]:
                display_frame = pil_puttext(display_frame, hint, (10, y_off), (200,200,200), 17)
                y_off += 22

            h, w = display_frame.shape[:2]
            cv2.line(display_frame, (w//3,0), (w//3,h), (255,255,255),2)
            cv2.line(display_frame, (2*w//3,0), (2*w//3,h), (255,255,255),2)
            cv2.line(display_frame, (0,h//3), (w,h//3), (255,255,255),2)
            cv2.line(display_frame, (0,2*h//3), (w,2*h//3), (255,255,255),2)

            cv2.imshow('AI Camera', display_frame)
            frame_count += 1

        else:  # 相册
            if not photo_list:
                black = np.zeros((480,640,3), dtype=np.uint8)
                black = pil_puttext(black, "没有照片", (200,200), (255,255,255), 30)
                black = pil_puttext(black, "按 L 返回", (200,240), (150,150,150), 20)
                cv2.imshow('AI Camera', black)
            else:
                img = cv2.imread(photo_list[current_photo_idx])
                if img is not None:
                    img = cv2.resize(img, (640,480))
                    img = pil_puttext(img, f"{current_photo_idx+1}/{len(photo_list)}", (10,10), (255,255,255), 18)
                    img = pil_puttext(img, "A 上一张 | D 下一张 | L 返回", (10,440), (200,200,200), 20)
                    cv2.imshow('AI Camera', img)

        key = cv2.waitKey(1) & 0xFF
        gpio_key = check_gpio_buttons()
        if gpio_key is not None:
            key = ord(gpio_key)

        if key == ord('q'): break
        elif key == ord('s') and view_mode=="camera":
            if raw_frame is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SAVE_DIR, f"photo_{ts}.jpg")
                cv2.imwrite(path, raw_frame)
                print(f"💾 {path}")
        elif key == ord('g'): show_spiral = not show_spiral
        elif key == ord('l'):
            if view_mode=="camera":
                view_mode="gallery"
                photo_list = sorted(glob.glob(os.path.join(SAVE_DIR,"*.jpg"))+
                                   glob.glob(os.path.join(SAVE_DIR,"*.png")))
                current_photo_idx = len(photo_list)-1 if photo_list else -1
            else:
                view_mode="camera"
        elif key == ord('a') and view_mode=="gallery":
            if photo_list: current_photo_idx = max(0, current_photo_idx-1)
        elif key == ord('d') and view_mode=="gallery":
            if photo_list: current_photo_idx = min(len(photo_list)-1, current_photo_idx+1)

    state.running = False
    ai_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    print("👋 退出")

if __name__ == '__main__':
    main()