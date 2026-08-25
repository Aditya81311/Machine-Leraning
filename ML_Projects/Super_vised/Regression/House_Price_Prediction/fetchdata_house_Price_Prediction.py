import os 
import tarfile 
from urllib import request

download_root = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
Housing_path = os.path.join("datasets","housing")
Housing_url = download_root + "datasets/housing/housing.tgz"

def fetch_data(housing_url = Housing_url,housing_path = Housing_path):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_data()