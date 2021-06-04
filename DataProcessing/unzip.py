import zipfile
import os
files = zipfile.ZipFile('TFLClassify.zip', 'r')
files.extractall()
files.close()