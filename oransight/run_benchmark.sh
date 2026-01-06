#!/bin/bash

# --- CẤU HÌNH ---
SCRIPT_NAME="main_benchmark.py"
# Tạo tên file log theo thời gian thực (Ví dụ: benchmark_result_20231025_143000.txt)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="benchmark_result_${TIMESTAMP}.txt"

# --- MÀU SẮC (Cho đẹp output màn hình) ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   ORAN-BENCH-13K AUTOMATION SCRIPT         ${NC}"
echo -e "${GREEN}============================================${NC}"
echo "Thời gian bắt đầu: $(date)"
echo "File log sẽ được lưu tại: $LOG_FILE"
echo ""

# 1. Kiểm tra môi trường
if [ ! -f "$SCRIPT_NAME" ]; then
    echo -e "${RED}❌ Lỗi: Không tìm thấy file python '$SCRIPT_NAME'.${NC}"
    echo "Hãy đảm bảo bạn đang đứng đúng thư mục chứa code."
    exit 1
fi

if [ ! -d "oran_specs" ]; then
    echo -e "${YELLOW}⚠️  Cảnh báo: Không tìm thấy thư mục 'oran_specs'.${NC}"
    echo "Mô hình sẽ chạy mà không có Knowledge Base (Điểm số sẽ thấp)."
else
    echo -e "${GREEN}✅ Đã tìm thấy thư mục 'oran_specs'.${NC}"
fi

if [ ! -d "benchmark" ]; then
    echo -e "${RED}❌ Lỗi: Không tìm thấy thư mục 'benchmark'.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Đã tìm thấy thư mục 'benchmark'.${NC}"
fi

# 2. Thực thi script Python
# Lệnh 'python3 -u' giúp unbuffer output (in ra ngay lập tức)
# Lệnh '| tee' giúp vừa in ra màn hình vừa ghi vào file log
# '2>&1' giúp bắt cả thông báo lỗi (stderr) vào log luôn

echo ""
echo -e "${YELLOW}🚀 Đang khởi chạy Benchmark...${NC}"
echo "--------------------------------------------"

python3 -u "$SCRIPT_NAME" 2>&1 | tee "$LOG_FILE"

# 3. Kiểm tra kết quả
# Lấy exit code của lệnh python (phần tử đầu tiên trong pipeline)
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "--------------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Hoàn tất thành công!${NC}"
    echo "Bạn có thể xem lại kết quả trong file: $LOG_FILE"
else
    echo -e "${RED}❌ Có lỗi xảy ra trong quá trình chạy (Exit Code: $EXIT_CODE).${NC}"
    echo "Vui lòng kiểm tra file log: $LOG_FILE"
fi