import cv2
import numpy as np

from pathlib import Path

VIDEO_PATH = Path('room_fhd.mp4')

def draw_keypoints(image, keypoints, show=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if show:
        for i in keypoints:
            x,y = i.ravel() 
            cv2.circle(image,(x,y),3,255,-1)
            
            cv2.imshow('frame',image)
            cv2.waitKey(1)

def draw_optical_flow(image, optical_flow):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color = (0,0,255)

    for i in optical_flow:
        x, y = i[0].ravel()
        u, v = i[1].ravel()

        start_point = (x,y) 
        end_point = (x+int(u), y+int(v))
        cv2.cv2.arrowedLine(image, start_point, end_point, color)
        
    cv2.imshow('frame',image)
    cv2.waitKey(1)

def in_margin(x, y, N, width, height):
    if (x-N//2 > 0) and (x-N//2+1 < width):
        if (y-N//2 > 0) and (x-N//2+1 < height):
            return True
    return False

def sum_N_window(Ia, x, y, N):
    return Ia[x-N//2:x+N//2+1, y-N//2:y+N//2+1].sum()



def calculate_matrix_thing(frame, prev_frame, keypoints, N=15):
    Ix = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    Iy = cv2.Sobel(frame, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    It = (prev_frame - frame)

    height, width = frame.shape

    optical_flow = []
    for k in keypoints:
        x, y = k.ravel()
        
        if not in_margin(x, y, N, width, height): continue

        IxIx = Ix * Ix 
        IyIy = Iy * Iy
        IxIy = Ix * Iy  
        IxIt = Ix * It
        IyIt = Iy * It

        A = np.array([sum_N_window(IxIx, x, y, N), sum_N_window(IxIy, x, y, N),
                      sum_N_window(IxIy, x, y, N), sum_N_window(IyIy, x, y, N)]).reshape(2,2)
        B = np.array([sum_N_window(IxIt, x, y, N),
                      sum_N_window(IyIt, x, y, N)]).reshape(2,1)
 
        try:
            inverse_A = np.linalg.inv(A)
        except:
            continue
        u_v = -np.matmul(inverse_A, B)
        u, v = (u_v).ravel()
        optical_flow.append([np.array([x,y]), np.array([u,v])])

    draw_optical_flow(frame, optical_flow)

    return optical_flow


def main():
    cap = cv2.VideoCapture(f'{VIDEO_PATH}')

    frame_n = 0
    prev_frame = []
    while cap.isOpened():
        print(f'On frame {frame_n}')
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret is False:
            break

        corners =  np.int0(cv2.goodFeaturesToTrack(gray, 200, 0.01, 1))
        draw_keypoints(frame, corners, False)

        if frame_n>0:
            calculate_matrix_thing(gray, prev_frame, corners)

        prev_frame = gray
        frame_n += 1

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()