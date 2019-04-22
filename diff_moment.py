import cv2
import numpy as np
import matplotlib.pyplot as plt
from dynamic_plotting import DynamicPlot

# configuration
number_feature_point = 50
move_threshold = 2
overflow_threshold = 25
idle_status_threshold = 0.2
status_history_depth = 3
frame_interval = 10  # ms


def main():
    dp = DynamicPlot(50)
    # init cv2 part
    camera = cv2.VideoCapture(0)
    width = int(camera.get(3))
    height = int(camera.get(4))
    # read first frame as reference
    ret, last_frame = camera.read()
    last_frame = cv2.flip(last_frame, 1)
    last_bw = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    last_cx = 0
    last_cy = 0
    while True:
        # break when press ESC
        if cv2.waitKey(frame_interval) & 0xff == 27:
            break
        ret, current_frame = camera.read()
        current_frame=cv2.flip(current_frame, 1)
        current_bw = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(current_bw, last_bw)
        kernel = np.ones((5, 5), np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
        moments = cv2.moments(diff)
        if moments['m00'] != 0:
            # normal
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            delta_cx = cx - last_cx
            delta_cy = cy - last_cy
        else:
            # no difference between two frames
            cx = cy = 0
            delta_cx = 0
            delta_cy = 0

        avg_delta = np.sum(diff) / (width * height)

        preview = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        cv2.circle(preview, (cx, cy), 10, (0, 0, 255), -1)
        cv2.imshow('Press ESC to stop...', preview)
        dp.add_data(avg_delta, 0, delta_cx, delta_cy)

        last_frame = current_frame.copy()
        last_bw = current_bw.copy()
        last_cx = cx
        last_cy = cy
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
