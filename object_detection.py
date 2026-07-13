import ultralytics as ut
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
#predict_class = [0,14]
img = 'avengers-infinity-war-cast.jpg'
#pred = model(img, classes = predict_class, conf = .4)

pred = model(img, conf = .4)

#count the number of objects class-wise

class_names = model.names
counts = {}

for cls in pred[0].boxes.cls:
    cls = int(cls)
    name = class_names[cls]
    counts[name] = counts.get(name, 0) + 1

print("Object Counts:")
for name, count in counts.items():
    print(f"{name}: {count}")


pred_img = pred[0].plot()

import cv2
cv2.imwrite('avengers2.jpg', pred_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
