"""
Virtual Painter — On-screen GUI selectable with selection gesture

Features added:
- On-screen GUI buttons (Undo, Redo, Save, Clear, Eraser) arranged vertically on the right
- Gesture selection: with Index+Middle up (selection mode) hover the selection rectangle center
  over a button; hold for `HOVER_THRESHOLD` seconds to activate the button action
- Palette selection at top still supported via the same hover/hold mechanism
- Visual feedback: hovered button highlights and a progress arc / bar fills while holding

Usage:
- Draw: index finger up
- Selection Mode: index + middle fingers up -> hover over GUI & hold to trigger
- Gesture brush size: distance between index tip (8) and thumb tip (4)
- Keyboard shortcuts remain: q=quit, c=clear, s=save, e=toggle eraser, z=undo, y=redo
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque

# ---------- Setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

# Canvas & state
canvas = None
px, py = 0, 0
brush_thickness = 12
draw_color = (255, 0, 255)  # default purple (BGR)
smooth_factor = 0.7

# Stroke storage for undo/redo
strokes = []
redo_stack = []
current_segments = None

thickness_buf = deque(maxlen=5)

# Palette
palette = [
    ((255, 0, 255), "Purple"),
    ((0, 0, 255), "Red"),
    ((0, 255, 0), "Green"),
    ((255, 0, 0), "Blue"),
    ((0, 0, 0), "Eraser")
]

# Timing / hover
HOVER_THRESHOLD = 0.6  # seconds to hold to activate
hover_target = None
hover_start = None
last_trigger_time = 0
TRIGGER_COOLDOWN = 0.5  # small cooldown to avoid immediate re-trigger

last_time = time.time()
fps = 0

# ---------- Helpers ----------
def ensure_canvas(frame):
    global canvas
    if canvas is None or canvas.shape != frame.shape:
        canvas = np.zeros_like(frame)

def save_output(frame, canvas, prefix="virtual_paint"):
    out = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    os.makedirs("captures", exist_ok=True)
    filename = f"captures/{prefix}_{int(time.time())}.png"
    cv2.imwrite(filename, out)
    return filename

def rebuild_canvas():
    """Redraw canvas from strokes list."""
    global canvas
    if canvas is None:
        return
    canvas[:] = 0
    for stroke in strokes:
        for seg in stroke:
            x1, y1, x2, y2, thickness, color = seg
            cv2.line(canvas, (x1, y1), (x2, y2), color, int(thickness))
            cv2.circle(canvas, (x2, y2), max(1, int(thickness/2)), color, -1)

# ---------- GUI Button Model ----------
class Button:
    def __init__(self, x, y, w, h, label, color=(50,50,50), action=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.color = color
        self.action = action
        self.hover_progress = 0.0  # 0..1

    def draw(self, frame, is_hovered=False, progress=0.0):
        # Background
        bg = tuple(int(min(255, c + 40)) for c in self.color) if is_hovered else self.color
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, cv2.FILLED)
        # Border
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (255,255,255), 2)
        # Label
        cv2.putText(frame, self.label, (self.x + 10, self.y + self.h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # Progress bar at bottom of button if hovered
        if is_hovered and progress > 0:
            pb_w = int(self.w * progress)
            cv2.rectangle(frame, (self.x, self.y + self.h - 8), (self.x + pb_w, self.y + self.h - 2), (0,200,0), cv2.FILLED)

    def contains(self, px, py):
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)

# ---------- Actions ----------
def action_undo():
    global strokes, redo_stack
    if strokes:
        redo_stack.append(strokes.pop())
        rebuild_canvas()

def action_redo():
    global strokes, redo_stack
    if redo_stack:
        strokes.append(redo_stack.pop())
        rebuild_canvas()

def action_save(frame):
    fn = save_output(frame, canvas)
    print("Saved:", fn)

def action_clear():
    global strokes, redo_stack, canvas
    strokes.clear()
    redo_stack.clear()
    if canvas is not None:
        canvas[:] = 0

def action_toggle_eraser():
    global draw_color
    if draw_color == (0,0,0):
        draw_color = (255,0,255)
    else:
        draw_color = (0,0,0)

# ---------- Build Buttons (right side vertical) ----------
def build_buttons(frame_shape):
    h, w = frame_shape[0], frame_shape[1]
    btn_w, btn_h = 140, 60
    margin = 20
    x = w - btn_w - margin
    # stack from top margin down
    y = margin
    buttons = []
    buttons.append(Button(x, y, btn_w, btn_h, "Undo (z)", color=(70,70,70), action=action_undo)); y += btn_h + 12
    buttons.append(Button(x, y, btn_w, btn_h, "Redo (y)", color=(70,70,70), action=action_redo)); y += btn_h + 12
    buttons.append(Button(x, y, btn_w, btn_h, "Save (s)", color=(60,120,60), action=lambda: action_save(current_frame_copy))); y += btn_h + 12
    buttons.append(Button(x, y, btn_w, btn_h, "Clear (c)", color=(120,60,60), action=action_clear)); y += btn_h + 12
    buttons.append(Button(x, y, btn_w, btn_h, "Eraser (e)", color=(40,40,40), action=action_toggle_eraser))
    return buttons

# ---------- Palette Drawing (top) ----------
def draw_palette(frame):
    side = 80
    margin = 10
    for i, (col, label) in enumerate(palette):
        x1 = margin + i * (side + margin)
        y1 = margin
        x2 = x1 + side
        y2 = y1 + side
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
        cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

def pick_palette_index(x):
    margin = 10
    side = 80
    for i in range(len(palette)):
        x1 = margin + i * (side + margin)
        x2 = x1 + side
        if x1 <= x <= x2:
            return i
    return None

# ---------- Main Loop ----------
buttons = None
current_frame_copy = None  # used by save action lambda closure

print("Controls: q=quit, c=clear, s=save, e=toggle eraser, z=undo, y=redo")
print("Selection: index+middle up -> hover over GUI and hold to activate")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    ensure_canvas(frame)
    if buttons is None:
        buttons = build_buttons(frame.shape)

    # keep a copy for actions that may need the current frame (save)
    current_frame_copy = frame.copy()

    # FPS
    now = time.time()
    fps = 1.0 / (now - last_time) if now != last_time else fps
    last_time = now

    # Draw palette UI
    draw_palette(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    index_up = False
    middle_up = False
    cx = cy = mx = my = tx = ty = None

    selection_active = False
    sel_x = sel_y = None

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm = hand_lms.landmark
            cx, cy = int(lm[8].x * w), int(lm[8].y * h)
            mx, my = int(lm[12].x * w), int(lm[12].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y

            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # compute brush thickness with thumb-index distance
            if tx is not None and cx is not None:
                dist = np.hypot(cx - tx, cy - ty)
                min_d, max_d = 20, 220
                mapped = np.interp(dist, [min_d, max_d], [4, 80])
                thickness_buf.append(mapped)
                brush_thickness = int(np.mean(thickness_buf))

            # Selection mode
            if index_up and middle_up:
                selection_active = True
                # selection rectangle center
                x1, y1 = min(cx, mx), min(cy, my)
                x2, y2 = max(cx, mx), max(cy, my)
                sel_x = (x1 + x2) // 2
                sel_y = (y1 + y2) // 2
                # draw selection rectangle & label
                cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)
                cv2.putText(frame, "Selection Mode", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                # visualize center
                cv2.circle(frame, (sel_x, sel_y), 6, (255,255,255), cv2.FILLED)
            else:
                selection_active = False

    # If in selection mode, check palette (top) and buttons (right)
    hovered = None
    hovered_type = None  # 'button' or 'palette'
    hovered_index = None

    if selection_active and sel_x is not None and sel_y is not None:
        # Check palette first (top)
        top_threshold = 120
        if sel_y < top_threshold:
            idx = pick_palette_index(sel_x)
            if idx is not None:
                hovered = idx
                hovered_type = 'palette'
                hovered_index = idx
        else:
            # Check buttons
            for b in buttons:
                if b.contains(sel_x, sel_y):
                    hovered = b
                    hovered_type = 'button'
                    break

    # Hover management + activation
    current_time = time.time()
    if hovered is not None:
        # same target as before?
        if hover_target is None or (hover_target != hovered):
            hover_target = hovered
            hover_start = current_time
        else:
            # continuing to hover
            elapsed = current_time - hover_start
            progress = min(1.0, elapsed / HOVER_THRESHOLD)
            # draw hover progress on appropriate UI element
            if hovered_type == 'button':
                hovered.draw(frame, is_hovered=True, progress=progress)
            else:
                # palette hover: draw a ring/progress near the palette item
                # compute palette rect center for visual
                side = 80
                margin = 10
                x1 = margin + hovered * (side + margin)
                y1 = margin
                cx_p = x1 + side // 2
                cy_p = y1 + side // 2
                # ring
                cv2.circle(frame, (cx_p, cy_p), side//2 + 6, (255,255,255), 2)
                # progress arc expressed as rectangle fill for simplicity
                pb_w = int(side * progress)
                cv2.rectangle(frame, (x1, y1 + side - 8), (x1 + pb_w, y1 + side - 2), (0,200,0), cv2.FILLED)

            # trigger if elapsed passes threshold and cooldown passed
            if elapsed >= HOVER_THRESHOLD and (current_time - last_trigger_time) > TRIGGER_COOLDOWN:
                # Activate
                last_trigger_time = current_time
                # perform action
                if hovered_type == 'button':
                    if hovered.action:
                        hovered.action()
                else:  # palette
                    chosen_color, label = palette[hovered]
                    draw_color = chosen_color  # pylint: disable=unused-variable
                    # note: drawing color here modifies local binding; to affect outer scope, we reassign below
                    # we'll set a flag to apply it after hover logic
                    apply_palette = (chosen_color, label)
                # reset hover
                hover_target = None
                hover_start = None
    else:
        # not hovering anything; reset hover state
        hover_target = None
        hover_start = None

    # If palette was chosen during activation, apply it (we used a local variable)
    if 'apply_palette' in locals():
        chosen_color, label = apply_palette
        draw_color = chosen_color

        # feedback text
        if label.lower() == "eraser":
            cv2.putText(frame, "Eraser Selected", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, f"{label} Selected", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        del apply_palette  # clear local flag

    # Draw buttons (non-hovered ones)
    for b in buttons:
        # If it's the current hovered button it has been drawn already with progress; avoid double-draw
        if hover_target is not None and hover_target == b:
            # already drawn above when hovered; skip drawing baseline
            continue
        b.draw(frame, is_hovered=False, progress=0)

    # ---------- Drawing Logic (index up only) ----------
    # NOTE: We keep the previous stroke logic: start/append segments in drawing mode
    if results.multi_hand_landmarks:
        # reuse variables from above; index_up indicates drawing mode when True and middle_up False
        if index_up and not middle_up:
            # drawing mode
            if current_segments is None:
                current_segments = []
                redo_stack.clear()

            # smoothing
            if px == 0 and py == 0:
                sx, sy = cx, cy
            else:
                sx = int(px * smooth_factor + cx * (1 - smooth_factor))
                sy = int(py * smooth_factor + cy * (1 - smooth_factor))

            cv2.circle(frame, (cx, cy), 8, draw_color, cv2.FILLED)

            seg_color = draw_color
            seg_thickness = brush_thickness
            if px != 0 or py != 0:
                cv2.line(canvas, (px, py), (cx, cy), seg_color, seg_thickness)
                cv2.circle(canvas, (cx, cy), max(1, int(seg_thickness/2)), seg_color, -1)
                current_segments.append((px, py, cx, cy, seg_thickness, seg_color))
            px, py = cx, cy
        else:
            # not drawing; finalize stroke if any
            if current_segments:
                if len(current_segments) > 0:
                    strokes.append(current_segments)
                current_segments = None
            px, py = 0, 0
    else:
        # no hand detected: finalize in-progress stroke
        if current_segments:
            if len(current_segments) > 0:
                strokes.append(current_segments)
            current_segments = None
        px, py = 0, 0

    # Compose final image with canvas overlay
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    out = cv2.add(frame_bg, canvas_fg)

    # UI text
    cv2.putText(out, f"Brush: {brush_thickness}", (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(out, f"FPS: {int(fps)}", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    mode_text = "Eraser" if draw_color == (0,0,0) else "Draw"
    cv2.putText(out, f"Mode: {mode_text}", (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(out, "Selection: hover & hold (index+middle up) to activate buttons", (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    cv2.imshow("Virtual Painter (GUI)", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        action_clear()
    elif key == ord('s'):
        action_save(current_frame_copy)
    elif key == ord('e'):
        action_toggle_eraser()
    elif key == ord('z'):
        action_undo()
    elif key == ord('y'):
        action_redo()
    elif key == ord('+') or key == ord('='):
        brush_thickness = min(200, brush_thickness + 2)
    elif key == ord('-') or key == ord('_'):
        brush_thickness = max(1, brush_thickness - 2)

cap.release()
cv2.destroyAllWindows()
