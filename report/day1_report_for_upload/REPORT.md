# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
6. Vì sao không được lấy `score` của model làm ground truth?


1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**. Bởi vì đây là mô hình Phân loại ảnh (Image Classification), nó gán nhãn duy nhất đại diện cho toàn bộ ngữ cảnh bức ảnh chứ không khoanh vùng riêng biệt từng thực thể đơn lẻ.
3. Danh sách tên lớp (class list) đi kèm với **taxonomy/checkpoint** có sẵn (ImageNet-1K), được định nghĩa trước khi huấn luyện mô hình.
4. Cần ghi cả 3 thông tin vì `class_id` giúp máy xử lý chính xác không lỗi chính tả, `class_name` giúp con người đọc hiểu trực quan, còn `taxonomy_name` xác định hệ thống phân loại đang dùng (tránh nhầm lẫn class 468 của ImageNet với các bộ dữ liệu khác như COCO hay Pascal VOC).
5. Guideline cần quy định rõ **Quy tắc ưu tiên (Priority Rule)**: ví dụ khi ảnh có nhiều chủ đề, ưu tiên chọn đối tượng chiếm diện tích lớn nhất ở tiền cảnh (foreground), hoặc chọn đối tượng trung tâm đại diện cho bối cảnh chung.
6. Không lấy `score` làm ground truth vì score là độ tin cậy chủ quan của mô hình (có sai số và nhầm lẫn), còn ground truth phải phản ánh sự thật khách quan tuyệt đối do con người gán nhãn và kiểm duyệt.




**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?


1. Chép một record: `instance_id`="traffic-001" ; `class_name`=bus ; `score`=0.925745 ; số điểm polygon=`polygon_point_count`=120 ; 3 điểm đầu của `polygon_xy`=[[148.0, 189.0], [147.0, 190.0], [145.0, 190.0]]
2. So với box, polygon cung cấp chính xác đường biên bao quanh từng pixel thực tế của đối tượng, giúp loại bỏ hoàn toàn phần pixel nền dư thừa.
3. Câu này nghĩa là `instance_id` là mã định danh duy nhất để phân biệt thực thể này với thực thể khác trong cùng một bức ảnh tại thời điểm gán nhãn hiện tại, chứ không biểu thị phân lớp chung (class ID) hay theo dấu xuyên suốt qua nhiều khung hình video (tracking ID).
4. **Quy tắc vẽ biên mask:** Đặt các điểm neo (vertices) dọc theo ranh giới thực của đối tượng, khoảng cách giữa các điểm từ 5-10 pixel, bo sát các đoạn cong mềm mại và không được vẽ lấn sang các vật thể đè phía trước.
5. Chỗ người đầu bếp đứng che một phần bếp lò (`oven`). Tôi sẽ dùng **guideline** vì đây là trường hợp che khuất chồng chéo (occlusion) điển hình trong gán nhãn, guideline sẽ quy định vẽ phần nhìn thấy của oven và cắt biên chính xác theo hình dáng người đứng trước.




Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn văn bản duy nhất đại diện cho toàn bộ ảnh (ví dụ: "traffic"). | Ảnh có cả xe buýt, xe taxi, người đi bộ, rất khó chọn ra một lớp duy nhất đại diện nếu không có độ ưu tiên rõ ràng. | Chọn lớp phù hợp nhất theo quy tắc ưu tiên trong tài liệu hướng dẫn (Guideline). | Kiểm tra xem nhãn được chọn có đúng lớp đại diện tối ưu nhất của bối cảnh không. |
| Phát hiện vật thể | Các cặp tọa độ hộp chữ nhật `[x_min, y_min, x_max, y_max]` kèm nhãn lớp. | Một số chiếc bát (`bowl`) bị chồng lên nhau hoặc đặt ở góc khuất thiếu sáng, rất khó vẽ hộp độc lập bám khít. | Vẽ các hộp giới hạn ôm sát từng vật thể riêng biệt bám sát đường biên thực ngoài cùng. | Kiểm tra độ khít của hộp giới hạn, loại bỏ các hộp bị trùng lặp hoặc bổ sung hộp bị gán thiếu. |
| Instance segmentation | Tập hợp các điểm đa giác `[[x1, y1], [x2, y2], ...]` bo theo ranh giới của từng thực thể. | Biên của chiếc muôi treo hoặc các kẽ hở nhỏ giữa tay và thân người đầu bếp rất phức tạp để đi nét đa giác chính xác. | Click chuột tỉ mỉ bao quanh viền thực của vật thể, phân tách rõ ràng các thực thể chạm nhau. | Kiểm tra độ chi tiết của đường biên, đảm bảo không có pixel rác của nền bị gộp nhầm vào mặt nạ (mask). |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?


1. Quy tắc bảo vệ dữ liệu: Chỉ làm việc và lưu trữ các tập dữ liệu công khai được cấp phép, tuyệt đối không tải lên dữ liệu bảo mật hoặc hình ảnh cá nhân lên môi trường Colab dùng chung.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi hoặc chứa thông tin nhạy cảm, tôi sẽ dừng ngay lập tức và báo cáo trực tiếp cho **Lab Coach hoặc Giảng viên phụ trách lớp**.




Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
