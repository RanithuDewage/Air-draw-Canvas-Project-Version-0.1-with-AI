import cv2
import numpy as np
import math
from datetime import datetime

# ---------------------------------
# CONFIG
# ---------------------------------
WIDTH, HEIGHT = 1280, 720
TOOLBAR_H = 112
BOX_HALF = 30

DEFAULT_BRUSH_THICKNESS = 6
MIN_BRUSH_THICKNESS = 2
MAX_BRUSH_THICKNESS = 20

DEFAULT_ERASER_THICKNESS = 28
MIN_ERASER_THICKNESS = 10
MAX_ERASER_THICKNESS = 60

MIN_AREA = 220
MAX_AREA = 5000
MIN_RADIUS = 7

MAX_TRACK_JUMP = 120
MAX_DRAW_JUMP = 70
SMOOTH_ALPHA = 0.45

MASK_AREA_LIMIT = 0.04
BUTTON_COOLDOWN = 10

# ---------------------------------
# SOFT PASTEL COLORS (BGR)
# ---------------------------------
PASTEL_BRUSHES = [
    {"name": "SKY",   "color": (255, 220, 185)},
    {"name": "BLUSH", "color": (220, 200, 255)},
    {"name": "MINT",  "color": (210, 242, 205)},
    {"name": "PEACH", "color": (190, 225, 255)},
    {"name": "LILAC", "color": (255, 220, 238)},
    {"name": "LEMON", "color": (190, 245, 255)},
    {"name": "ROSE",  "color": (215, 185, 245)},
    {"name": "ICE",   "color": (255, 245, 220)},
]

ERASER_COLOR = (255, 255, 255)

# ---------------------------------
# HELPERS
# ---------------------------------
def distance(p1, p2):
    return int(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def contour_center(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def hue_mask(hsv, hue, tol, s_low, v_low):
    low = (hue - tol) % 180
    high = (hue + tol) % 180

    if low <= high:
        mask = cv2.inRange(hsv, (low, s_low, v_low), (high, 255, 255))
    else:
        mask1 = cv2.inRange(hsv, (0, s_low, v_low), (high, 255, 255))
        mask2 = cv2.inRange(hsv, (low, s_low, v_low), (179, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

    return mask


def calibrate_from_roi(roi_hsv):
    pixels = roi_hsv.reshape(-1, 3)

    strong = pixels[(pixels[:, 1] >= 125) & (pixels[:, 2] >= 65)]
    if len(strong) < 120:
        return None

    hist = np.bincount(strong[:, 0], minlength=180)
    hue = int(np.argmax(hist))

    diffs = np.abs(((strong[:, 0].astype(np.int16) - hue + 90) % 180) - 90)
    near = strong[diffs <= 10]

    if len(near) < 80:
        return None

    s_med = int(np.median(near[:, 1]))
    v_med = int(np.median(near[:, 2]))

    if s_med < 120:
        return None

    return {
        "hue": hue,
        "tol": 10,
        "s_low": max(110, s_med - 50),
        "v_low": max(50, v_med - 60)
    }


def choose_best_contour(contours, last_center):
    best_cnt = None
    best_center = None
    best_score = -1e9

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        center = contour_center(cnt)
        if center is None:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        short_side = max(1, min(w, h))
        long_side = max(w, h)
        aspect = long_side / short_side

        if aspect > 3.5:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = max(1.0, cv2.contourArea(hull))
        solidity = area / hull_area
        circularity = (4 * math.pi * area) / (peri * peri)

        score = (circularity * 3.0) + (solidity * 2.0) + (area * 0.002)

        if last_center is not None:
            d = distance(center, last_center)
            if d > MAX_TRACK_JUMP:
                score -= 100
            else:
                score -= d * 0.03

        if score > best_score:
            best_score = score
            best_cnt = cnt
            best_center = center

    return best_cnt, best_center


def text_color_for_bg(bgr):
    b, g, r = bgr
    brightness = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if brightness > 160 else (255, 255, 255)


def get_current_style(current_color_idx, eraser_mode, brush_thickness, eraser_thickness):
    if eraser_mode:
        return {
            "color": ERASER_COLOR,
            "thickness": eraser_thickness,
            "name": "ERASER"
        }
    return {
        "color": PASTEL_BRUSHES[current_color_idx]["color"],
        "thickness": brush_thickness,
        "name": PASTEL_BRUSHES[current_color_idx]["name"]
    }


def draw_all_strokes(canvas, strokes):
    for stroke in strokes:
        pts = stroke["points"]
        color = stroke["color"]
        thickness = stroke["thickness"]

        if len(pts) == 1:
            cv2.circle(canvas, pts[0], max(1, thickness // 2), color, -1)

        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, thickness)


def render_paint_canvas(strokes):
    canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    draw_all_strokes(canvas, strokes)
    return canvas


def create_toolbar_buttons():
    buttons = []

    top_items = [
        {"kind": "clear", "label": "CLEAR", "fill": (30, 30, 30)},
        {"kind": "undo",  "label": "UNDO",  "fill": (70, 70, 70)},
        {"kind": "save",  "label": "SAVE",  "fill": (90, 130, 90)},
        {"kind": "eraser","label": "ERASE", "fill": (220, 220, 220)},
        {"kind": "minus", "label": "-",     "fill": (120, 120, 120)},
        {"kind": "plus",  "label": "+",     "fill": (120, 120, 120)},
    ]

    top_w = 110
    top_h = 42
    gap = 8
    total_top_w = len(top_items) * top_w + (len(top_items) - 1) * gap
    start_x_top = (WIDTH - total_top_w) // 2
    y_top = 6

    x = start_x_top
    for item in top_items:
        item["rect"] = (x, y_top, x + top_w, y_top + top_h)
        buttons.append(item)
        x += top_w + gap

    color_w = 110
    color_h = 42
    total_color_w = len(PASTEL_BRUSHES) * color_w + (len(PASTEL_BRUSHES) - 1) * gap
    start_x_colors = (WIDTH - total_color_w) // 2
    y_colors = 58

    x = start_x_colors
    for i, brush in enumerate(PASTEL_BRUSHES):
        buttons.append({
            "kind": "color",
            "label": brush["name"],
            "fill": brush["color"],
            "color_idx": i,
            "rect": (x, y_colors, x + color_w, y_colors + color_h)
        })
        x += color_w + gap

    return buttons


def draw_eraser_icon(frame, rect):
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2 + 2

    pts = np.array([
        [cx - 18, cy + 8],
        [cx - 4,  cy - 8],
        [cx + 18, cy + 6],
        [cx + 4,  cy + 22]
    ], np.int32)

    cv2.fillConvexPoly(frame, pts, (250, 250, 250))
    cv2.polylines(frame, [pts], True, (80, 80, 80), 2)
    cv2.line(frame, (cx - 10, cy + 15), (cx + 10, cy - 2), (190, 190, 190), 2)


def draw_toolbar(frame, buttons, current_color_idx, eraser_mode,
                 brush_thickness, eraser_thickness,
                 draw_enabled, calibrated, tracking, status_text):
    for btn in buttons:
        x1, y1, x2, y2 = btn["rect"]
        fill = btn["fill"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)

        selected = False
        if btn["kind"] == "eraser" and eraser_mode:
            selected = True
        elif btn["kind"] == "color" and (not eraser_mode) and btn["color_idx"] == current_color_idx:
            selected = True

        border_color = (255, 255, 255) if selected else (235, 235, 235)
        border_thick = 3 if selected else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)

        if btn["kind"] == "eraser":
            draw_eraser_icon(frame, btn["rect"])
            cv2.putText(frame, "ERASE", (x1 + 18, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
        else:
            tc = text_color_for_bg(fill)
            font_scale = 0.62 if len(btn["label"]) <= 5 else 0.55
            cv2.putText(frame, btn["label"], (x1 + 12, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, tc, 2)

    cv2.putText(frame, f"DRAW: {'ON' if draw_enabled else 'OFF'}",
                (20, HEIGHT - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    cv2.putText(frame, "CALIBRATED" if calibrated else "NOT CALIBRATED",
                (180, HEIGHT - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (0, 255, 0) if calibrated else (0, 0, 255), 2)

    cv2.putText(frame, "TRACKING" if tracking else "NO OBJECT",
                (410, HEIGHT - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (0, 255, 0) if tracking else (0, 0, 255), 2)

    cv2.putText(frame, f"BRUSH:{brush_thickness}px  ERASER:{eraser_thickness}px",
                (620, HEIGHT - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    cv2.putText(frame,
                "k=calibrate  u=undo  e=eraser  [+/-]=size  c=clear  s=save  d=draw  q=quit",
                (20, HEIGHT - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, status_text, (20, HEIGHT - 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)


# ---------------------------------
# CAMERA
# ---------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working ❌")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

kernel = np.ones((5, 5), np.uint8)
buttons = create_toolbar_buttons()

# ---------------------------------
# STATE
# ---------------------------------
strokes = []
active_stroke = None

current_color_idx = 0
eraser_mode = False
draw_enabled = True

brush_thickness = DEFAULT_BRUSH_THICKNESS
eraser_thickness = DEFAULT_ERASER_THICKNESS

calibrated = False
calib = None

smooth_center = None
button_cooldown = 0
status_text = "Put a bright BLUE/GREEN marker in the box, then press K"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not captured ❌")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    live_output = frame.copy()
    paintwindow = render_paint_canvas(strokes)
    mask_display = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    tracking = False
    center = None
    radius = 0
    circ_x, circ_y = 0, 0

    cx, cy = WIDTH // 2, HEIGHT // 2
    x1, y1 = cx - BOX_HALF, cy - BOX_HALF
    x2, y2 = cx + BOX_HALF, cy + BOX_HALF

    # ---------------------------------
    # TRACKING
    # ---------------------------------
    if calibrated and calib is not None:
        mask = hue_mask(hsv, calib["hue"], calib["tol"], calib["s_low"], calib["v_low"])
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        mask_display = mask.copy()
        white_ratio = cv2.countNonZero(mask) / float(WIDTH * HEIGHT)

        if white_ratio <= MASK_AREA_LIMIT:
            contours_info = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            best_cnt, raw_center = choose_best_contour(contours, smooth_center)

            if best_cnt is not None:
                ((circ_x, circ_y), radius) = cv2.minEnclosingCircle(best_cnt)

                if radius >= MIN_RADIUS:
                    if smooth_center is None:
                        smooth_center = raw_center
                    else:
                        if distance(raw_center, smooth_center) > MAX_TRACK_JUMP:
                            smooth_center = raw_center
                        else:
                            smooth_center = (
                                int(SMOOTH_ALPHA * raw_center[0] + (1 - SMOOTH_ALPHA) * smooth_center[0]),
                                int(SMOOTH_ALPHA * raw_center[1] + (1 - SMOOTH_ALPHA) * smooth_center[1])
                            )

                    center = smooth_center
                    tracking = True
                    status_text = "Tracking OK"
                else:
                    smooth_center = None
                    status_text = "Object too small"
            else:
                smooth_center = None
                status_text = "No valid marker found"
        else:
            smooth_center = None
            tracking = False
            status_text = "Mask too noisy - recalibrate with bright marker"
    else:
        smooth_center = None

    if not tracking:
        active_stroke = None

    if button_cooldown > 0:
        button_cooldown -= 1

    # ---------------------------------
    # DRAW / TOOLBAR ACTIONS
    # ---------------------------------
    if tracking and center is not None:
        cv2.circle(live_output, (int(circ_x), int(circ_y)), int(radius), (0, 255, 255), 2)

        cursor_style = get_current_style(current_color_idx, eraser_mode, brush_thickness, eraser_thickness)
        cursor_color = (170, 170, 170) if eraser_mode else cursor_style["color"]
        cursor_size = max(8, cursor_style["thickness"] // 2)

        cv2.circle(live_output, center, cursor_size, cursor_color, 2)
        cv2.circle(live_output, center, 4, (0, 0, 255), -1)

        # Toolbar area
        if center[1] <= TOOLBAR_H:
            active_stroke = None

            if button_cooldown == 0:
                for btn in buttons:
                    bx1, by1, bx2, by2 = btn["rect"]
                    if bx1 <= center[0] <= bx2 and by1 <= center[1] <= by2:
                        kind = btn["kind"]

                        if kind == "clear":
                            strokes.clear()
                            active_stroke = None
                            status_text = "Canvas cleared"

                        elif kind == "undo":
                            if strokes:
                                strokes.pop()
                                active_stroke = None
                                status_text = "Last stroke removed"
                            else:
                                status_text = "Nothing to undo"

                        elif kind == "save":
                            save_img = render_paint_canvas(strokes)
                            filename = f"air_draw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            cv2.imwrite(filename, save_img)
                            status_text = f"Saved: {filename}"
                            print(f"Saved as {filename} ✅")

                        elif kind == "eraser":
                            eraser_mode = True
                            active_stroke = None
                            status_text = "Eraser selected"

                        elif kind == "minus":
                            if eraser_mode:
                                eraser_thickness = max(MIN_ERASER_THICKNESS, eraser_thickness - 2)
                                status_text = f"Eraser size: {eraser_thickness}"
                            else:
                                brush_thickness = max(MIN_BRUSH_THICKNESS, brush_thickness - 1)
                                status_text = f"Brush size: {brush_thickness}"
                            active_stroke = None

                        elif kind == "plus":
                            if eraser_mode:
                                eraser_thickness = min(MAX_ERASER_THICKNESS, eraser_thickness + 2)
                                status_text = f"Eraser size: {eraser_thickness}"
                            else:
                                brush_thickness = min(MAX_BRUSH_THICKNESS, brush_thickness + 1)
                                status_text = f"Brush size: {brush_thickness}"
                            active_stroke = None

                        elif kind == "color":
                            current_color_idx = btn["color_idx"]
                            eraser_mode = False
                            active_stroke = None
                            status_text = f"Brush: {PASTEL_BRUSHES[current_color_idx]['name']}"

                        button_cooldown = BUTTON_COOLDOWN
                        break

        else:
            if draw_enabled:
                style = get_current_style(current_color_idx, eraser_mode, brush_thickness, eraser_thickness)

                if active_stroke is None:
                    active_stroke = {
                        "points": [center],
                        "color": style["color"],
                        "thickness": style["thickness"],
                        "name": style["name"]
                    }
                    strokes.append(active_stroke)
                else:
                    last_point = active_stroke["points"][-1]
                    if distance(last_point, center) <= MAX_DRAW_JUMP:
                        active_stroke["points"].append(center)
                    else:
                        active_stroke = {
                            "points": [center],
                            "color": style["color"],
                            "thickness": style["thickness"],
                            "name": style["name"]
                        }
                        strokes.append(active_stroke)

    # ---------------------------------
    # DRAW SAVED STROKES
    # ---------------------------------
    draw_all_strokes(live_output, strokes)
    paintwindow = render_paint_canvas(strokes)

    # ---------------------------------
    # CALIBRATION BOX
    # ---------------------------------
    cv2.rectangle(live_output, (x1, y1), (x2, y2), (255, 255, 255), 2)

    if not calibrated:
        cv2.putText(live_output, "Hold BRIGHT BLUE/GREEN marker inside the box and press K",
                    (WIDTH // 2 - 270, HEIGHT // 2 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    draw_toolbar(
        live_output,
        buttons,
        current_color_idx,
        eraser_mode,
        brush_thickness,
        eraser_thickness,
        draw_enabled,
        calibrated,
        tracking,
        status_text
    )

    cv2.imshow("Live Drawing", live_output)
    cv2.imshow("Paint", paintwindow)
    cv2.imshow("Mask", mask_display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        strokes.clear()
        active_stroke = None
        status_text = "Canvas cleared"

    elif key == ord('u'):
        if strokes:
            strokes.pop()
            active_stroke = None
            status_text = "Last stroke removed"
        else:
            status_text = "Nothing to undo"

    elif key == ord('d'):
        draw_enabled = not draw_enabled
        active_stroke = None
        status_text = "Draw ON" if draw_enabled else "Draw OFF"

    elif key == ord('e'):
        eraser_mode = True
        active_stroke = None
        status_text = "Eraser selected"

    elif key == ord('s'):
        save_img = render_paint_canvas(strokes)
        filename = f"air_draw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, save_img)
        status_text = f"Saved: {filename}"
        print(f"Saved as {filename} ✅")

    elif key == ord('+') or key == ord('='):
        if eraser_mode:
            eraser_thickness = min(MAX_ERASER_THICKNESS, eraser_thickness + 2)
            status_text = f"Eraser size: {eraser_thickness}"
        else:
            brush_thickness = min(MAX_BRUSH_THICKNESS, brush_thickness + 1)
            status_text = f"Brush size: {brush_thickness}"
        active_stroke = None

    elif key == ord('-') or key == ord('_'):
        if eraser_mode:
            eraser_thickness = max(MIN_ERASER_THICKNESS, eraser_thickness - 2)
            status_text = f"Eraser size: {eraser_thickness}"
        else:
            brush_thickness = max(MIN_BRUSH_THICKNESS, brush_thickness - 1)
            status_text = f"Brush size: {brush_thickness}"
        active_stroke = None

    elif key == ord('k'):
        roi = hsv[y1:y2, x1:x2]
        result = calibrate_from_roi(roi)

        if result is None:
            calibrated = False
            calib = None
            smooth_center = None
            active_stroke = None
            status_text = "Calibration FAILED - put bright BLUE/GREEN marker in the box"
            print("Calibration failed ❌")
        else:
            calibrated = True
            calib = result
            smooth_center = None
            active_stroke = None
            status_text = f"Calibrated hue={calib['hue']} | move marker to draw"
            print("Calibration success ✅")
            print(calib)

cap.release()
cv2.destroyAllWindows()