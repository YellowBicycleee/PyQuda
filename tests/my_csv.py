import csv
import os
import numpy as np

# 检查文件存在，并且超过200行
def file_exists(file_path):
    return os.path.isfile(file_path) and os.stat(file_path).st_size != 0

def read_csv(file_name):
  const_ = []
  lambda_ = []
  plaquette_ = []
  with open(file_name, "r", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      const_.append(float(row['const']))
      lambda_.append(float(row['lambda']))
      plaquette_.append(float(row['plaquette']))
  return const_, lambda_, plaquette_

def write_csv(file_name, data):
  with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

# read last column of each line in file
def get_last_column(file_name) -> np.array:
  data = []
  with open(file_name, 'r') as file:
    for line in file:
      # 使用split()函数以空格为分隔符分割每行，并取最后一个元素
      data.append(float(line.split()[-1]))
  return np.array(data)


def generate_data (color, const: np.array, log_path) :
    
  result = []
  result.append(['color', 'const', 'lambda', 'plaquette'])
  for i, const_elem in enumerate(const):
    file_path = f'{log_path}/res_const_{const[i]}.txt'
    
    data = get_last_column(file_path)
    if data.size > 0:
      # Nc, const, beta, lambda = 3/const
      result.append([color, const_elem, 3 / const_elem, get_last_column(file_path)[-1]])
    else :
      print('data is empty')
      exit(-1)
  return result


def generate_csv (color: int, Ns : int, Nt : int, lst : list, csv_path, log_path) -> list:
  const = np.array(lst)
  result = generate_data(color, const, log_path=log_path)
  write_csv(csv_path, result)
  return lst