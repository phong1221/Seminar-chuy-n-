# Mini Transformer cho Phân loại Cảm xúc Văn bản

Đây là bộ khung đồ án theo đúng cấu trúc yêu cầu trong đề.

## Cấu trúc thư mục

```text
MSSV_HoTen_DoAn/
├── data/
│   └── sentiment_raw.csv
├── data_utils.py
├── model.py
├── train.py
├── visualize.py
├── requirements.txt
└── README.md
```

## Chạy theo thứ tự

### 1) Cài thư viện
```bash
pip install -r requirements.txt
```

### 2) Tiền xử lý dữ liệu
```bash
python data_utils.py --max_len 20 --show_stats
```

### 3) Điền TODO trong `model.py`, sau đó tự kiểm tra
```bash
python model.py
```

Kỳ vọng khi điền đúng:
- scaled_dot_product_attention ... PASSED
- SelfAttention ... PASSED
- FeedForwardNetwork ... PASSED
- TransformerEncoderBlock ... PASSED

### 4) Huấn luyện
```bash
python train.py
python train.py --run_all
python train.py --d_model 128 --d_ff 256
```

### 5) Visualize attention
```bash
python visualize.py
python visualize.py --model results/model_Transformer_d128_ff256.pt
python visualize.py --sentence "this film is absolutely terrible"
```

## Ghi chú cho sinh viên
- Chỉ cần điền các phần `# TODO` trong `model.py`.
- Không đổi tên hàm và tham số.
- Dữ liệu đã có sẵn cột `split`.
- Bộ dữ liệu này là dữ liệu mô phỏng cân bằng 3 lớp để phục vụ học Transformer.
