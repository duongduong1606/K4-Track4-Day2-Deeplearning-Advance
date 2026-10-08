# DeepWeeds Lab Day 2 submission

MSSV: `2A202602498`

Họ tên không dấu: `Duong Duong`

Notebook Colab: https://colab.research.google.com/github/duongduong1606/K4-Track4-Day2-Deeplearning-Advance/blob/main/code/lab_day2.ipynb

## Môi trường đã chạy

- `Python 3.13.15`
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `timm==1.0.30`
- `numpy==2.5.2`
- `pandas==3.0.6`
- `scikit-learn==1.9.1`
- `Pillow==12.3.0`
- `matplotlib==3.11.2`
- `openpyxl==3.1.5`
- `thop==0.1.1-2209072238`

## Thứ tự tái lập

1. Cài `code/requirements.txt` và chuẩn bị DeepWeeds fold 0.
2. Chạy pipeline checks, screening queue và inference sweep chỉ trên validation.
3. Chạy final training với seed 42, 123, 2026.
4. Mở khóa final test manifest sau khi đóng băng inference spec.
5. Chạy `eval.py score`, `eval.py grade`, rồi tạo `results.xlsx` và báo cáo.

Chi tiết cấu hình từng run nằm trong `results.xlsx`, `code/experiment_plan.py` và prediction CSV.
