import cv2
import mediapipe as mp
import time
import pyautogui
import numpy as np

# =======================================================
# 摄像头与运行模式配置
# =======================================================
CAMERA_INDEX = 0
HEADLESS_MODE = False
STATUS_PRINT_INTERVAL = 1.0
DEBUG_PRINT_INTERVAL = 0.5
STABLE_FRAME_COUNT = 5

# =======================================================
# 鼠标控制参数
# =======================================================
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CLICK_COOLDOWN = 1.0
TRACKING_SMOOTHING = 0.6

# =======================================================
# 状态定义
# =======================================================
STATE_TRACKING = "TRACKING"
STATE_LOCKED = "LOCKED"

GESTURE_ONE = "ONE"
GESTURE_TWO = "TWO"
GESTURE_FIST = "FIST"
GESTURE_PALM = "PALM"
GESTURE_UNKNOWN = "UNKNOWN"

pyautogui.FAILSAFE = False


def is_thumb_open(landmarks):
    """根据拇指关键点横向位置判断拇指是否伸开。"""
    thumb_tip_x = landmarks[4].x
    thumb_ip_x = landmarks[3].x
    return abs(thumb_tip_x - thumb_ip_x) > 0.04


def is_finger_up(landmarks, tip_id, pip_id):
    """判断除拇指外手指是否伸直。"""
    return landmarks[tip_id].y < landmarks[pip_id].y


def analyze_fingers(hand_landmarks):
    """分析当前手部各手指状态，供手势识别与调试输出共用。"""
    landmarks = hand_landmarks.landmark

    finger_states = {
        "thumb_open": is_thumb_open(landmarks),
        "index_up": is_finger_up(landmarks, 8, 6),
        "middle_up": is_finger_up(landmarks, 12, 10),
        "ring_up": is_finger_up(landmarks, 16, 14),
        "pinky_up": is_finger_up(landmarks, 20, 18)
    }
    return finger_states


def recognize_gesture_from_states(finger_states):
    """根据手指状态识别目标手势。"""
    index_up = finger_states["index_up"]
    middle_up = finger_states["middle_up"]
    ring_up = finger_states["ring_up"]
    pinky_up = finger_states["pinky_up"]
    thumb_open = finger_states["thumb_open"]

    # 数字 1：仅食指伸直
    if index_up and not middle_up and not ring_up and not pinky_up:
        return GESTURE_ONE

    # 数字 2：食指与中指伸直
    if index_up and middle_up and not ring_up and not pinky_up:
        return GESTURE_TWO

    # 握拳：四指全部弯曲
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return GESTURE_FIST

    # 巴掌：四指全部伸直，且拇指明显张开
    if index_up and middle_up and ring_up and pinky_up and thumb_open:
        return GESTURE_PALM

    return GESTURE_UNKNOWN


def get_palm_center(hand_landmarks, frame_width, frame_height):
    """取腕部(0) + 四指根部(5,9,13,17) 五点平均作为掌心坐标。"""
    indices = [0, 5, 9, 13, 17]
    x_sum = sum(hand_landmarks.landmark[i].x for i in indices)
    y_sum = sum(hand_landmarks.landmark[i].y for i in indices)
    cx = int((x_sum / 5) * frame_width)
    cy = int((y_sum / 5) * frame_height)
    return cx, cy


def map_to_screen(palm_x, palm_y, frame_width, frame_height):
    """绝对映射：画面坐标 → 屏幕坐标，夹紧到屏幕边界。"""
    screen_x = np.interp(palm_x, [0, frame_width], [0, SCREEN_WIDTH])
    screen_y = np.interp(palm_y, [0, frame_height], [0, SCREEN_HEIGHT])
    return int(screen_x), int(screen_y)


def format_bool_state(value):
    """将布尔状态转换为便于终端阅读的中文描述。"""
    return "伸展" if value else "弯曲"


def print_finger_debug(finger_states, current_gesture):
    """打印当前手指状态与候选手势，帮助定位识别规则问题。"""
    print(
        "检测到手 | "
        f"拇指:{'张开' if finger_states['thumb_open'] else '收拢'} | "
        f"食指:{format_bool_state(finger_states['index_up'])} | "
        f"中指:{format_bool_state(finger_states['middle_up'])} | "
        f"无名指:{format_bool_state(finger_states['ring_up'])} | "
        f"小指:{format_bool_state(finger_states['pinky_up'])} | "
        f"候选手势:{current_gesture}"
    )


def main():
    # 1. 初始化 MediaPipe 手部识别环境
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # 2. 打开摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"系统级错误: 摄像头打开失败 -> 编号 {CAMERA_INDEX}")
        hands.close()
        return

    # 3. 初始化状态机变量
    current_state = STATE_TRACKING

    last_gesture = GESTURE_UNKNOWN
    stable_count = 0
    last_click_time = 0
    prev_cx, prev_cy = 0, 0
    last_debug_print_time = 0

    print("系统进入运行状态：五指张开 = 追踪鼠标，握拳 = 锁定鼠标")
    print("手势 1 = 左键单击，手势 2 = 右键单击（锁定模式下有效）")
    print(f"当前摄像头编号: {CAMERA_INDEX}, 屏幕分辨率: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    if HEADLESS_MODE:
        print("当前运行模式: 无窗口模式，按 Ctrl+C 可手动终止程序")
    else:
        print("当前运行模式: 摄像头窗口模式，按 q 键退出程序")

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success or image is None:
                print("运行警告: 摄像头读取失败，循环结束")
                break

            image = cv2.flip(image, 1)
            frame_h, frame_w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            current_gesture = GESTURE_UNKNOWN
            current_time = time.time()

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # 绘制手部关键点
                if not HEADLESS_MODE:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                finger_states = analyze_fingers(hand_landmarks)
                current_gesture = recognize_gesture_from_states(finger_states)

                if (current_time - last_debug_print_time) >= DEBUG_PRINT_INTERVAL:
                    print_finger_debug(finger_states, current_gesture)
                    last_debug_print_time = current_time
            else:
                if (current_time - last_debug_print_time) >= DEBUG_PRINT_INTERVAL:
                    print("未检测到手")
                    last_debug_print_time = current_time

            # 4. 连续多帧稳定确认机制
            if current_gesture == last_gesture and current_gesture != GESTURE_UNKNOWN:
                stable_count += 1
            elif current_gesture != GESTURE_UNKNOWN:
                last_gesture = current_gesture
                stable_count = 1
            else:
                last_gesture = GESTURE_UNKNOWN
                stable_count = 0

            # ============ 状态机 ============

            if current_state == STATE_TRACKING:
                # 手掌追踪模式：掌心 → 鼠标移动
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    cx, cy = get_palm_center(hand_landmarks, frame_w, frame_h)

                    # 指数移动平均平滑
                    if prev_cx != 0 and prev_cy != 0:
                        cx = int(prev_cx * TRACKING_SMOOTHING + cx * (1 - TRACKING_SMOOTHING))
                        cy = int(prev_cy * TRACKING_SMOOTHING + cy * (1 - TRACKING_SMOOTHING))
                    prev_cx, prev_cy = cx, cy

                    screen_x, screen_y = map_to_screen(cx, cy, frame_w, frame_h)
                    pyautogui.moveTo(screen_x, screen_y)

                # 检测到握拳 → 切换到锁定模式
                if stable_count >= STABLE_FRAME_COUNT and last_gesture == GESTURE_FIST:
                    print("切换到锁定模式")
                    current_state = STATE_LOCKED
                    stable_count = 0
                    last_gesture = GESTURE_UNKNOWN

            elif current_state == STATE_LOCKED:
                # 锁定模式：鼠标不跟随
                # 检测到巴掌 → 切回追踪模式
                if stable_count >= STABLE_FRAME_COUNT and last_gesture == GESTURE_PALM:
                    print("切换到追踪模式")
                    current_state = STATE_TRACKING
                    prev_cx, prev_cy = 0, 0  # 重置平滑状态
                    stable_count = 0
                    last_gesture = GESTURE_UNKNOWN

                # 检测到点击手势（仅锁定模式下生效，带冷却）
                if stable_count >= STABLE_FRAME_COUNT:
                    if (current_time - last_click_time) >= CLICK_COOLDOWN:
                        if last_gesture == GESTURE_ONE:
                            pyautogui.click(button="left")
                            print("左键单击")
                            last_click_time = current_time
                            stable_count = 0
                            last_gesture = GESTURE_UNKNOWN
                        elif last_gesture == GESTURE_TWO:
                            pyautogui.click(button="right")
                            print("右键单击")
                            last_click_time = current_time
                            stable_count = 0
                            last_gesture = GESTURE_UNKNOWN

            # 画面显示
            if not HEADLESS_MODE:
                # 叠加状态信息
                state_text = f"状态: {current_state}"
                gesture_text = f"手势: {current_gesture}"
                cv2.putText(image, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.putText(image, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                cv2.imshow('Gesture Mouse Control', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        hands.close()
        print("系统已安全退出")


if __name__ == "__main__":
    main()
