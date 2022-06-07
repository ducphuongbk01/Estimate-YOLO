import glob
import os

data_path = glob.glob("C:\\Users\\phuon\\Downloads\\Compressed\\2200-2399\\ano\\*.txt")
direct = "C:\\Users\\phuon\\Downloads\\Compressed\\2200-2399\\ano\\"

print(len(data_path))

for i, path in enumerate(data_path):
    new_path = direct + str(i+2200) + ".txt"
    os.rename(path, new_path)
    print("Renamed image " + new_path)