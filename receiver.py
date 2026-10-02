"""Ground-side viewer for the UDP video stream. Run on the receiving PC:  python receiver.py 5000
Then in the dashboard: Settings -> Stream -> enter this PC's IP and the port -> Connect."""
import socket, sys, cv2, numpy as np
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 5000))
print("Waiting for stream... press Q in the window to quit")
while True:
    data, _ = s.recvfrom(65536)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None: cv2.imshow("BAS stream", img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
