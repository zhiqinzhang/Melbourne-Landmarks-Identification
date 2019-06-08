import os

# __file__ refers to the file settings.py
MOBILENET_ROOT = os.path.dirname(os.path.abspath(__file__))

dir_list = MOBILENET_ROOT.split("/")[:-2]
ROOT = ""
for dir in dir_list:
    ROOT += "/" + str(dir)
ROOT = ROOT[1:]
print(ROOT)

OUTPUT_ROOT = os.path.join(MOBILENET_ROOT,'output')
