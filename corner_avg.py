import cv2
import numpy as np
from dynamic_plotting import DynamicPlot
import csv

# configuration
# capture_src = r"C:\Users\tomzh\Downloads\01\001.mkv"  # filename
capture_src = 0  # 0 - built-in camera
is_flipped = True  # recommended for camera
number_feature_point = 75
move_threshold = 1
overflow_threshold = 20
idle_status_threshold = 0.2
status_history_depth = 3
frame_interval = 10  # ms

export_as_csv = False
csv_filename = "sample.csv"


def sign(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0


def main():
    dp = DynamicPlot(50)
    # init cv2 part
    camera = cv2.VideoCapture(capture_src)
    width = int(camera.get(3))
    height = int(camera.get(4))
    last_avg_cp_x = 0
    last_avg_cp_y = 0
    status_x_history = [0] * status_history_depth
    status_y_history = [0] * status_history_depth
    status_history_cursor = 0
    # read first frame as reference
    ret, last_frame = camera.read()
    frame_index = 0
    f = open(csv_filename, "w")
    f_csv = csv.writer(f)
    f_csv.writerow(["frame", "l1_diff_sum", "l1_has_corners", "l1_avg_x", "l1_avg_y", "l2_motion_state", "l2_delta_x",
                    "l2_delta_y", "l3_result"])
    while ret:
        # break when press ESC
        if cv2.waitKey(frame_interval) & 0xff == 27:
            break
        camera.read()
        ret, current_frame = camera.read()
        # absolute difference
        img = cv2.absdiff(current_frame, last_frame)
        # flip on x-axis
        if is_flipped:
            img = cv2.flip(img, 1)
        # polarization
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # morph open to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        if is_flipped:
            preview = cv2.flip(current_frame, 1)
        else:
            preview = current_frame.copy()
        '''
        #
        # Too slow, don't use it!
        # Just subtract 255 by average value. Don't modify image itself.
        #
        # reverse color
        for i in range(height):
            for j in range(width):
                img[i, j] = (255 - img[i, j])
        '''

        # avoid scene switch
        avg_delta = np.sum(img) / (width * height)
        if avg_delta > overflow_threshold:
            motion_status = 1
        elif avg_delta < move_threshold:
            motion_status = -1
        else:
            motion_status = 0

        corners = cv2.goodFeaturesToTrack(img, number_feature_point, 0.01, 10)
        if corners is not None:
            has_corners = True
            corners = np.int0(corners)
            x_sum = 0
            y_sum = 0
            for pt in corners:
                x, y = pt.ravel()
                x_sum += x
                y_sum += y
            current_avg_cp_x = x_sum / len(corners)
            current_avg_cp_y = y_sum / len(corners)
        else:
            has_corners = False
            current_avg_cp_x = 0
            current_avg_cp_y = 0

        if motion_status == 0:
            delta_avg_cp_x = current_avg_cp_x - last_avg_cp_x
            delta_avg_cp_y = current_avg_cp_y - last_avg_cp_y
            if abs(delta_avg_cp_x) > abs(delta_avg_cp_y):
                status_x_history[status_history_cursor] += sign(delta_avg_cp_x)
                status_y_history[status_history_cursor] = 0
            else:
                status_y_history[status_history_cursor] += sign(delta_avg_cp_y)
                status_x_history[status_history_cursor] = 0
            # draw corner points
            for i in corners:
                x, y = i.ravel()
                cv2.circle(preview, (x, y), 3, (0, 255, 255), -1)
        else:
            delta_avg_cp_x = 0
            delta_avg_cp_y = 0
            status_x_history[status_history_cursor] = 0
            status_y_history[status_history_cursor] = 0
        last_avg_cp_x = current_avg_cp_x
        last_avg_cp_y = current_avg_cp_y
        status_history_cursor += 1
        if status_history_cursor == status_history_depth:
            status_history_cursor = 0
        # plot update
        dp.add_data(avg_delta, motion_status, delta_avg_cp_x, delta_avg_cp_y)

        status_avg_x = sum(status_x_history) / status_history_depth
        status_avg_y = sum(status_y_history) / status_history_depth
        # draw text
        if abs(status_avg_x) < idle_status_threshold and abs(status_avg_y) < idle_status_threshold:
            final_text = "IDLE"
        elif abs(status_avg_x) > abs(status_avg_y):
            if status_avg_x > 0:
                final_text = "RIGHT"
            else:
                final_text = "LEFT"
        else:
            if status_avg_y > 0:
                final_text = "DOWN"
            else:
                final_text = "UP"
        cv2.putText(preview, final_text, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 8)
        # display
        cv2.imshow('Press ESC to stop...', preview)
        # write to file
        f_csv.writerow(
            [frame_index, avg_delta, has_corners, current_avg_cp_x, current_avg_cp_y, motion_status, delta_avg_cp_x,
             delta_avg_cp_y, final_text])
        # update last_frame
        last_frame = current_frame.copy()
        # increase frame_index
        frame_index += 1
    camera.release()
    cv2.destroyAllWindows()
    f.close()


if __name__ == '__main__':
    main()
