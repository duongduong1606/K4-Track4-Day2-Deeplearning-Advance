# DeepWeeds implementation

Thư mục này là bản hoàn chỉnh của `starter/`; không sửa `eval.py` và không sửa các CSV split.

## Cấu hình nền

- Fold 0, ảnh 224, batch 32, seed 42.
- ImageNet pretrained, fine-tune toàn bộ.
- AdamW: LR backbone `1e-4`, head `1e-3`, weight decay `0.05` (norm/bias của backbone không decay).
- 10 epoch, warmup 1 epoch rồi cosine, AMP.
- CE, augmentation `RandomResizedCrop + horizontal flip`.
- Checkpoint tốt nhất theo validation macro-F1; hòa giữ epoch sớm hơn.
- Test mặc định tắt và file test đã tồn tại sẽ không bị ghi đè.

## Chuẩn bị

Dùng môi trường Python riêng (khuyến nghị Python 3.11/3.12), cài PyTorch có CUDA phù hợp với máy,
sau đó cài các gói trong `requirements.txt`. Dữ liệu phải có dạng:

```text
data/images/*.jpg
data/labels/labels.csv
data/labels/train_subset0.csv
data/labels/val_subset0.csv
data/labels/test_subset0.csv
```

## Kiểm tra

```powershell
python -m unittest code.test_implementation -v
python -m unittest discover -s tests -v
```

## Chạy baseline

```powershell
python code/train.py --set exp_id=T00 backbone=resnet50 seed=42
```

Kết quả được lưu ở `runs/T00/seed42/`, prediction validation ở `predictions/`, và biểu đồ ở
`curves/`. Thêm `save_test_predictions=true` **chỉ sau khi đã chốt cấu hình hoàn toàn trên val**.

`experiment_plan.py` chứa sáu backbone, các ablation một-thay-đổi-mỗi-lần và ba seed chung kết.
Không chạy tự động toàn bộ để tránh vô tình tiêu tốn GPU hoặc mở test trước thời điểm quy định.

## Tự động hóa Google Colab

Đặt trên Drive:

```text
MyDrive/DeepWeedsLab/data/images.zip
MyDrive/DeepWeedsLab/data/labels/{labels,train_subset0,val_subset0,test_subset0}.csv
```

`colab_automation.py` cung cấp:

- `stage_dataset(...)`: kiểm tra MD5, copy ZIP và giải nén vào `/content` để tránh train trực tiếp trên Drive.
- `create_screening_manifest(...)`: tạo queue validation-only, không có quyền test.
- `run_queue(...)`: chạy tuần tự, bỏ qua run hoàn tất và resume `last.pt` sau khi runtime bị ngắt.
- `create_final_manifest(...)`: tạo queue baseline/final ba seed ở trạng thái khóa.
- `unlock_final_manifest(...)`: mở test bằng một thao tác tường minh kèm ghi chú lựa chọn từ validation.

Mỗi epoch lưu `last.pt` nguyên tử, giữ `best.pt` theo val macro-F1, đồng bộ checkpoint, history và
status lên `MyDrive/DeepWeedsLab/artifacts/`. Checkpoint chứa model, optimizer, scheduler, AMP scaler,
EMA, RNG của Python/NumPy/PyTorch/CUDA và trạng thái generator của DataLoader.

Trong notebook, đặt `RUN_SCREENING_QUEUE=True` để chạy tối đa ba run mỗi phiên. Các phiên sau chạy
lại cùng cell sẽ tự tiếp tục. Final test yêu cầu đồng thời: final manifest đã mở khóa và
`allow_test=True`; prediction test đã tồn tại sẽ không bị ghi đè.

Notebook Colab tự clone repository từ remote `origin`. Vì vậy phải commit và push thư mục `code/`
trước khi mở notebook trên Colab; nếu không, Colab chỉ thấy phiên bản cũ trên GitHub.
