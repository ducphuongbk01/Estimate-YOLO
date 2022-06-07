import os
import glob
import cv2
import numpy as np

path_input = "./data/val_anotation"
path_data = "./data/val/"
path_output = "./data/val_anotation_newFormat"

def check(path_txt):
        name = path_data + path_txt.split(".")[1].split("\\")[-1] + ".png"
        if os.path.isfile(name):
                return True
        return False

classes = None
with open("./yolo.names", 'r') as f: # Thay đổi địa chỉ của fiell yolo.names tại đây
    classes = [line.strip() for line in f.readlines()]

for path_txt in glob.glob(path_input + "/*.txt"):
    # print(path_txt.split(".")[1].split("\\")[-1])
    # print("\n")
    # print(path_data + path_txt.split(".")[1].split("\\")[-1])
    # break
    no=[0,0,0,0,0]
    #print(path_txt)
    if check(path_txt):
        image = cv2.imread(path_data + path_txt.split(".")[1].split("\\")[-1] + ".png")
        h,w = image.shape[:2]
        output=""
        with open(path_txt, "r") as stream:
            for line in stream.readlines():
                line = line.strip()
                coord = line.split(" ")
                a = classes[int(coord[0])]
                num = int(line.split(' ',1)[0])
                no[num]+=1
                coord = [float(i) for i in coord[1:]]
                box = np.array(coord) * np.array([w,h,w,h])
                (centerX, centerY, width, height) = box.astype('int')
                left = centerX - int(width/2)
                top = centerY - int(height/2)
                right = centerX + int(width/2)
                bottom = centerY + int(height/2)
                out1 = "{} {} {} {} {}\n".format(a, left, top, right, bottom)
                output +=out1
        name_output = path_txt.split("\\")[-1]
        print(name_output)
        with open(path_output + '/' + name_output, 'w') as stream_out:
            stream_out.write(output)
        stream.close()
        stream_out.close()