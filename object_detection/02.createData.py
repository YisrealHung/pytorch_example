import os
import shutil

# 存放圖片與標註檔的資料夾名稱
path = "dataset"
# 總類別
labels = ["sonar", "rfid"]

img_folder = "{}/JPEGImages".format(path)
xml_folder = "{}/Annotations".format(path)
lod_folder = "{}/ImageSets/Main".format(path)
lab_folder = "{}".format(path)



try:
    os.makedirs(img_folder)
except:
    pass

try:
    os.makedirs(xml_folder)
except:
    pass

try:
    os.makedirs(lod_folder)
except:
    pass

file1 = open('{}/test.txt'.format(lod_folder),'w+')
file2 = open('{}/val.txt'.format(lod_folder),'w+')
file3 = open('{}/train.txt'.format(lod_folder),'w+')
file4 = open('{}/trainval.txt'.format(lod_folder),'w+')
file5 = open('{}/labels.txt'.format(lab_folder),'w+')

for i in labels:
    file5.write("{}\n".format(i))
file5.close()

for i in os.listdir(path):
    print(i)
    if i[-4:] == '.jpg':
        shutil.move("{}/{}".format(path, i), img_folder)
    elif i[-4:] == '.xml':
        shutil.move("{}/{}".format(path, i), xml_folder)
        file1.write("{}\n".format(i[:-4]))
        file2.write("{}\n".format(i[:-4]))
        file3.write("{}\n".format(i[:-4]))
        file4.write("{}\n".format(i[:-4]))

file1.close()
file2.close()
file3.close()
file4.close()
