import zipfile

files = zipfile.ZipFile("TFLClassify.zip", "r")
files.extractall()
files.close()
