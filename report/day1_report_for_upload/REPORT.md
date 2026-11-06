# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

---

## 1. Phân loại ảnh

1. **Thông tin Record Rank-1 (Sample `traffic`):**
   - `class_id`: 468
   - `class_name`: cab
   - `rank`: 1
   - `score`: 0.510915
   - `taxonomy_name`: ImageNet-1K

2. **Giải thích phạm vi dự đoán:**
   Record này đại diện cho **toàn bộ bức ảnh**. Trong tác vụ Phân loại ảnh (Image Classification), mô hình không xác định vị trí hay khoanh vùng từng đối tượng đơn lẻ mà chỉ đưa ra một nhãn duy nhất đại diện cho tổng thể ngữ cảnh/bối cảnh của ảnh.

3. **Nguồn gốc danh sách tên lớp (Class List):**
   Danh sách này đi kèm trực tiếp với **taxonomy/checkpoint** định sẵn (ở đây là bộ ImageNet-1K), được xác định trước khi huấn luyện mô hình chứ không phải do mô hình tự ý sinh ra.

4. **Tầm quan trọng của việc ghi đủ thông tin phân loại:**
   - `class_id`: Giúp máy tính truy vấn, xử lý nhất quán, tránh lỗi sai lệch chữ viết hoặc dịch thuật.
   - `class_name`: Giúp lập trình viên và người dùng đọc hiểu trực quan.
   - `taxonomy_name`: Tránh xung đột nhãn giữa các bộ dữ liệu khác nhau (ví dụ: lớp 468 trong ImageNet có thể mang ý nghĩa hoàn toàn khác trong các bộ nhãn khác).

5. **Quy tắc gán nhãn khi ảnh có nhiều chủ đề (Guideline):**
   Guideline cần quy định rõ **Quy tắc ưu tiên (Priority Rule)**. Ví dụ: Ưu tiên chọn đối tượng chiếm diện tích lớn nhất ở tiền cảnh (foreground), hoặc đối tượng nằm ở vùng trung tâm (center) của ảnh đại diện cho chủ đề chính.

6. **Lý do không dùng score của model làm ground truth:**
   Model score chỉ phản ánh độ tin cậy chủ quan (confidence) mang tính xác suất của mô hình tại thời điểm suy luận (vẫn có sai số, nhiễu). Ground truth phải là sự thật khách quan (ground truth tuyệt đối) do con người gán nhãn, kiểm duyệt và xác thực chuẩn xác.

---

## 2. Phát hiện vật thể

1. **Thông tin Record được chọn (Sample `kitchen`):**
   - `class_name`: person
   - `score`: 0.912557
   - `bbox_xyxy`: [93.17, 187.95, 223.01, 320.91]
   - `bbox_width`: 129.84 px
   - `bbox_height`: 132.96 px

2. **Giải thích tọa độ hộp giới hạn (Bounding Box):**
   Điểm góc trên-bên trái của hộp giới hạn nằm ở tọa độ `(93.17, 187.95)` pixel và điểm góc dưới-bên phải nằm ở tọa độ `(223.01, 320.91)` pixel. Hộp này bao quanh thực thể với độ rộng `129.84` pixel và chiều cao `132.96` pixel.

3. **Số lượng dự đoán theo các ngưỡng khác nhau:**
   - Ngưỡng **0.20**: Phát hiện được **17** vật thể (bao gồm cả các vật nhỏ hoặc mờ như spoon, bottle, cup, potted plant).
   - Ngưỡng **0.35**: Phát hiện được **11** vật thể.
   - Ngưỡng **0.60**: Chỉ phát hiện được **6** vật thể rõ ràng nhất (person, bowl, oven).

4. **Ảnh hưởng khi hạ thấp ngưỡng tin cậy đối với Reviewer:**
   Khi hạ ngưỡng, số lượng hộp dự đoán sai (False Positive) tăng lên rõ rệt. Người Reviewer sẽ mất nhiều thời gian và công sức hơn để thủ công rà soát, lọc và xóa bỏ các hộp rác này.

5. **Quy tắc vẽ hộp giới hạn chuẩn (Bounding Box Rule):**
   Vẽ hộp bao khít ranh giới ngoài cùng của đối tượng (dung sai tối đa 2 pixel), không được cắt lẹm vào các bộ phận của đối tượng và hạn chế tối đa việc để thừa quá nhiều khoảng trống nền xung quanh.

6. **Quyết định gán nhãn cho vật thể bị che khuất/cắt mép:**
   Trên ảnh `kitchen`, chiếc nồi và chảo treo phía sau bị chồng chéo che khuất một phần. Chúng ta nên quy định trường hợp này trong **Guideline** (ví dụ: vẫn vẽ hộp đầy đủ nếu phần nhìn thấy được > 30% diện tích gốc) thay vì đưa lên Escalation. Điều này giúp tối ưu hóa thời gian và chỉ dùng Escalation cho những lỗi dị thường hoàn toàn mới.

---

## 3. Phân đoạn theo từng đối tượng

1. **Thông tin Record được chọn (Sample `traffic`):**
   - `instance_id`: "traffic-001"
   - `class_name`: bus
   - `score`: 0.925745
   - Số điểm đa giác (`polygon_point_count`): 120
   - 3 điểm tọa độ đầu tiên: `[[148.0, 189.0], [147.0, 190.0], [145.0, 190.0]]`

2. **Ưu điểm của Polygon so với Bounding Box:**
   Polygon cung cấp thông tin chi tiết khớp chính xác theo từng điểm ranh giới của đối tượng, giúp loại bỏ hoàn toàn các phần pixel nền không thuộc đối tượng bên trong hộp chữ nhật.

3. **Ý nghĩa của `instance_id`:**
   `instance_id` là mã định danh tạm thời nhằm phân biệt các cá thể thực thể khác nhau trong cùng một ảnh ở lượt xử lý hiện tại. Nó không mang tính chất phân loại chung (`class_id`) và không có tính năng theo dõi liên tục qua chuỗi khung hình video (`tracking_id`).

4. **Quy tắc vẽ biên đa giác (Polygon Mask Rule):**
   Đặt các điểm neo dọc sát theo đường ranh giới thực của đối tượng, khoảng cách trung bình từ 5-10 pixel, tăng mật độ điểm neo tại các đoạn cong phức tạp để đảm bảo mask mượt mà và ôm khít đối tượng.

5. **Giải quyết biên mờ/vật che khuất chồng chéo:**
   Khu vực người đầu bếp đứng che khuất một phần bếp lò (`oven`). Đây là trường hợp che phủ (occlusion) phổ biến nên cần được chuẩn hóa trong **Guideline** (ví dụ: chỉ gán nhãn đa giác cho phần nhìn thấy thực tế của bếp lò, đường biên cắt dọc theo tà tạp dề của người đầu bếp).

---

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Một nhãn văn bản duy nhất đại diện cho toàn bộ ảnh (ví dụ: "traffic"). | Ảnh chứa nhiều loại phương tiện đan xen (bus, cab, người đi bộ), khó xác định chủ thể duy nhất nếu thiếu quy tắc ưu tiên. | Chọn một lớp phù hợp nhất bám sát theo độ ưu tiên trong Guideline. | Đảm bảo lớp được chọn đại diện chính xác nhất cho ngữ cảnh toàn bức ảnh. |
| **Phát hiện vật thể** | Các cặp tọa độ hộp chữ nhật `[x_min, y_min, x_max, y_max]` kèm nhãn lớp. | Một số chiếc bát (`bowl`) chồng lấp lên nhau hoặc bị che tối ở góc bếp lò, khó vẽ hộp độc lập tách biệt. | Vẽ hộp chữ nhật bao bọc sát biên ngoài cùng của từng đối tượng độc lập. | Kiểm tra độ ôm khít, rà soát loại bỏ các hộp trùng lặp và bổ sung các hộp bị sót. |
| **Instance segmentation** | Tập hợp các điểm đa giác `[[x1, y1], [x2, y2], ...]` bao quanh ranh giới thực tế của từng cá thể. | Biên của chiếc muôi treo nhỏ hay khoảng trống kẽ tay người đầu bếp rất khó đi nét đa giác chính xác. | Click chuột tỉ mỉ bao quanh viền thực, phân tách rõ các mặt nạ của các đối tượng chạm sát nhau. | Kiểm tra độ chi tiết của đường biên, đảm bảo không bị gộp lẫn pixel rác của nền vào mặt nạ đối tượng. |

---

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu cá nhân:**
   Chỉ làm việc và lưu trữ các tập dữ liệu công khai được cấp phép, tuyệt đối không tải lên dữ liệu bảo mật hoặc hình ảnh cá nhân lên môi trường Colab dùng chung.

2. **Quy trình báo cáo dữ liệu ngoài phạm vi:**
   Nếu phát hiện hình ảnh không đúng phạm vi hoặc chứa thông tin nhạy cảm, tôi sẽ dừng tác vụ ngay lập tức và báo cáo trực tiếp cho **Lab Coach hoặc Giảng viên phụ trách**.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền đầy đủ và định dạng đẹp)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã chuẩn bị sẵn link repository để nộp trên LMS
