import cv2
import numpy as np
import matplotlib.pyplot as plt
from dynamic_plotting import DynamicPlot

# configuration
number_feature_point = 50
move_threshold = 2
overflow_threshold = 25
idle_status_threshold = 0.4
status_history_depth = 3
frame_interval = 2  # ms

motion_vector_learning_rate = 0.35

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def main():
    dp = DynamicPlot(50)
    # init cv2 part
    camera = cv2.VideoCapture(0)
    width = int(camera.get(3))
    height = int(camera.get(4))
    last_avg_cp_x = 0
    last_avg_cp_y = 0
    delta_avg_cp_x = 0
    delta_avg_cp_y = 0
    status_x_history = [0] * status_history_depth
    status_y_history = [0] * status_history_depth
    status_history_cursor = 0
    # read first frame as reference
    ret, last_frame = camera.read()
    current_corners = None
    last_corners = None
    has_last_corners = False
    current_gray = None
    last_gray = None

    lt_mv_x = 0
    lt_mv_y = 0
    while True:
        # break when press ESC
        if cv2.waitKey(frame_interval) & 0xff == 27:
            break
        camera.read()
        ret, current_frame = camera.read()
        # absolute difference
        img = cv2.absdiff(current_frame, last_frame)
        # flip on x-axis
        img = cv2.flip(img, 1)
        # erode to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # polarization
        img = cv2.medianBlur(img, 5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        current_gray = img

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
        motion_status = 0
        if avg_delta > overflow_threshold:
            motion_status = 1
        elif avg_delta < move_threshold:
            motion_status = -1
        else:
            motion_status = 0

        current_corners = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)
        mv_x = 0
        mv_y = 0
        if current_corners is not None:
            if has_last_corners:
                p1, st, err = cv2.calcOpticalFlowPyrLK(last_gray, current_gray, last_corners, None, **lk_params)
                # Select good points
                good_new = p1[st == 1]
                good_old = last_corners[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mv_x += a - c
                    mv_y += b - d
                l = len(good_new)
                if l != 0:
                    mv_x = mv_x / l * 50
                    mv_y = mv_y / l * 50
            has_last_corners = True
        else:
            has_last_corners = False

        lt_mv_x = motion_vector_learning_rate * mv_x + (1 - motion_vector_learning_rate) * lt_mv_x
        lt_mv_y = motion_vector_learning_rate * mv_y + (1 - motion_vector_learning_rate) * lt_mv_y

        if current_corners is not None:
            last_corners = current_corners.copy()
        last_gray = current_gray.copy()
        last_frame = current_frame.copy()
        dp.add_data(avg_delta, motion_status, lt_mv_x, lt_mv_y)
        # display
        cv2.imshow('Press ESC to stop...', img)
        # update last_frame
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
