import cv2

# 使用 DirectShow 后端（cv2.CAP_DSHOW）打开摄像头，设备索引一般为 0
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("无法打开摄像头，请检查驱动和设备连接")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧，退出...")
        break

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

cap.release()
cv2.destroyAllWindows()