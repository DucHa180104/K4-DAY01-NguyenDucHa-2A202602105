# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
>- Record hạng 1: `class_id: 468`, `class_name: cab`, `rank: 1`, `score: 0.510915`, `taxonomy_name: ImageNet-1K`.
- Record này mô tả toàn ảnh như thế nào?
>- Record mô tả prediction của model ở cấp toàn ảnh: cab là nhãn được xếp hạng cao nhất cho sample traffic. Record không xác định vị trí hoặc số lượng từng vật thể trong ảnh vì đây là image classification, không phải object detection.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
>- Class list được người xây dựng taxonomy/dataset và nhóm huấn luyện model định nghĩa trước. Checkpoint `yolo11n-cls.pt` được train với taxonomy ImageNet-1K nên chỉ có thể xếp hạng các nhãn thuộc danh sách này, thay vì tự tạo ra nhãn mới.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
>- Cần giữ `class_id` để máy xử lý và đối chiếu dữ liệu ổn định, `class_name` để con người hiểu nhãn, và `taxonomy_name` để xác định ID/tên lớp thuộc bộ nhãn nào. Nhờ đó tránh nhầm lẫn khi các dataset hoặc taxonomy dùng mã số hay cách gọi lớp khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
>- Nếu ảnh có nhiều chủ thể, guideline cần quy định cách chọn nhãn chính cho toàn ảnh, ví dụ ưu tiên vật thể liên quan trực tiếp đến mục tiêu dữ liệu, vật thể nổi bật hoặc chiếm diện tích lớn hơn. Guideline cũng cần nêu cách xử lý khi nhiều chủ thể có mức ưu tiên tương đương: dùng multi-label nếu hệ thống hỗ trợ, chọn theo thứ tự ưu tiên đã định nghĩa, hoặc chuyển reviewer/escalation khi ảnh mơ hồ.
- Vì sao model score không phải ground truth?
>- Model score là mức độ tin cậy nội bộ của model đối với prediction của nó, được sinh ra khi model xử lý ảnh. Score không phải ground truth vì model vẫn có thể tự tin nhưng dự đoán sai. Ground truth là nhãn chuẩn do con người hoặc dataset đã được kiểm tra và xác nhận theo guideline tạo ra.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
>- Một record: `class_name: person`, `score: 0.912625`, `bbox_xyxy: [385.33, 69.24, 498.92, 348.92]`, `bbox_width: 113.58`, `bbox_height: 279.68`.
- Diễn giải vị trí box bằng lời: 
>- Box `person` bắt đầu tại khoảng `(385, 69)` ở phía trên-bên phải ảnh và kết thúc tại khoảng `(499, 349)` ở phía dưới-bên phải. Box bao quanh người đứng ở khu vực giữa-phải của ảnh bếp.
- So sánh số prediction ở hai threshold:
>- Với sample `kitchen`, ở threshold `0.35` model giữ lại 11 prediction. Khi tăng threshold lên `0.60`, chỉ còn 6 prediction vì 5 box có score dưới 0.60 bị loại. Threshold cao hơn làm giảm số box được hiển thị và lưu trong output.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
>- Khi threshold giảm từ `0.60` xuống `0.35`, số prediction của sample `kitchen` tăng từ 6 lên 11. Độ bao phủ prediction tăng vì model giữ thêm các box có score thấp, nhờ đó có thể giảm nguy cơ bỏ sót vật thể. Tuy nhiên reviewer phải kiểm tra thêm 5 box và có thể gặp nhiều prediction sai hơn. Ngược lại, threshold `0.60` giảm khối lượng review nhưng có nguy cơ loại bỏ các vật thể thật có score thấp.
- Đề xuất một quy tắc box chặt:
>- Quy tắc box chặt: Box phải bao phủ toàn bộ phần nhìn thấy của đúng một vật thể thuộc class cần gán, bám sát các biên ngoài của vật thể nhưng không cắt mất phần vật thể nhìn thấy và không bao gồm quá nhiều nền hoặc vật thể bên cạnh. Với vật bị che khuất, chỉ box phần còn nhìn thấy, trừ khi guideline của dự án quy định khác.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
>- Với object bị che khuất hoặc cắt mép ảnh, guideline cần quy định có gán nhãn hay không, mức phần nhìn thấy tối thiểu để gán, box chỉ bao phần nhìn thấy hay ước lượng toàn bộ vật thể, và cách phân biệt bị che với bị cắt bởi biên ảnh. Nếu không xác định được class, biên vật thể hoặc phần nhìn thấy quá ít, annotator cần chuyển reviewer/escalation thay vì tự đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
>- Một record: `instance_id: kitchen-001`, `class_name: person`, `score: 0.899318`, `polygon_point_count: 348`, `polygon_xy` bắt đầu bằng `[[446, 70], [445, 71], [444, 71], [443, 72], [442, 72], ...]`.
- Polygon bổ sung chi tiết gì so với box?
>- Polygon bổ sung đường biên chi tiết theo hình dạng của từng instance bằng nhiều điểm `polygon_xy`, trong khi box chỉ dùng bốn tọa độ để tạo hình chữ nhật bao quanh vật thể. Vì vậy polygon phân biệt sát hơn phần thuộc về vật thể với nền hoặc vật thể bên cạnh, đặc biệt với vật có hình dạng không vuông.
- `instance_id` dùng để làm gì và không phải loại ID nào?
>- `instance_id` dùng để định danh riêng từng vật thể cụ thể trong một ảnh, giúp liên kết polygon, class, thuộc tính và kết quả QC của đúng instance đó. Ví dụ `kitchen-001` là một người cụ thể trong ảnh bếp. Đây không phải `class_id` dùng chung cho mọi vật thuộc cùng một lớp, không phải `coco_image_id`, và cũng không phải tracking ID theo cùng một vật thể qua nhiều frame video.
- Đề xuất một quy tắc biên mask:
>- Quy tắc biên mask:  Mask phải bám sát ranh giới phần nhìn thấy của đúng một instance, bao gồm mọi pixel thuộc vật thể và không lấn sang nền hoặc vật thể đang che/phía bên cạnh. Với vùng không nhìn rõ, annotator không tự suy đoán biên mà làm theo guideline hoặc chuyển reviewer.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
>- Với vùng mờ, các vật thể tiếp xúc nhau hoặc bị che khuất, guideline cần quy định biên mask phải dừng ở đâu, có tách thành các instance riêng không, chỉ mask phần nhìn thấy hay ước lượng phần bị che, và mức độ mờ tối thiểu để vẫn gán nhãn. Nếu không xác định được class hoặc ranh giới giữa các vật thể, annotator cần chuyển reviewer/escalation thay vì tự đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cấp ảnh thuộc taxonomy, gồm class_id và class_name; guideline quy định single-label hay multi-label. | Ảnh có nhiều chủ thể nên khó chọn nhãn chính; định nghĩa lớp có thể mơ hồ. | Quan sát toàn ảnh, gán nhãn theo taxonomy và guideline; escalation nếu ảnh mơ hồ. | Nhãn có đúng taxonomy, đúng quy tắc ưu tiên và nhất quán với ảnh tương tự không. |
| Phát hiện vật thể | Mỗi object có class_id, class_name và box bbox_xyxy. | Thiếu box, box trùng, sai class, box lỏng/chặt, vật bị che hoặc cắt mép. | Khoanh box cho từng vật thể thuộc phạm vi và gán đúng class. | Có bỏ sót/trùng object không; class đúng không; box có bám sát phần vật thể nhìn thấy không. |
| Instance segmentation | Mỗi instance có instance_id, class và mask/polygon gồm các điểm polygon_xy. |Mask lấn nền, thiếu vật thể, dính/tách nhầm instance; vùng mờ hoặc che khuất không rõ.  | Vẽ polygon/mask sát biên phần vật thể nhìn thấy, tách từng instance. | Mask có sát biên, lẫn nền/vật khác không; các instance tiếp xúc hoặc che khuất có được tách đúng không. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
>- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng, tải lên hoặc chia sẻ ảnh và output trong phạm vi được cấp quyền; không đưa dữ liệu nội bộ, dữ liệu có thông tin cá nhân hoặc dữ liệu chưa được phê duyệt lên GitHub công khai hay công cụ AI bên ngoài.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
>- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý và báo cho giảng viên, Lab Coach hoặc đầu mối phụ trách dữ liệu/bảo mật của chương trình để được hướng dẫn.

## 6. Danh sách bằng chứng

- [X ] `classification_predictions.json`
- [X ] `detection_predictions.json`
- [X ] `segmentation_predictions.json`
- [X ] `IMAGE_ATTRIBUTION.md`
- [X ] `visuals/classification_top5.png`
- [X ] `visuals/detection_predictions.png`
- [X ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [X ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
