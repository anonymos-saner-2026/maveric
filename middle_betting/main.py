import requests
import time
from datetime import datetime
from openai import OpenAI

# ================= CẤU HÌNH HỆ THỐNG =================
ODDS_API_KEY = "d169a9006ace263cc944f59d029e15c2"

# Key Yescale bạn cung cấp
YESCALE_API_KEY = "sk-AOzQMlsMqmhCbXzCAOOOCkFuOGi9Yx4741EpvrsdWpceYdNM" 
YESCALE_BASE_URL = "https://api.yescale.io/v1"
AI_MODEL = "gpt-4o" 

# Cấu hình ngưỡng
MIDDLE_BET_THRESHOLD = 12.0  # Chỉ gợi ý Middle nếu chấp > 12 điểm

# Khởi tạo Client
client = OpenAI(api_key=YESCALE_API_KEY, base_url=YESCALE_BASE_URL)

def get_nba_odds():
    """Lấy danh sách các trận đấu và kèo hiện tại"""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us,eu',
        'markets': 'spreads',
        'oddsFormat': 'decimal'
    }
    try:
        response = requests.get(url, params=params)
        return response.json() if response.status_code == 200 else []
    except Exception as e:
        print(f"[Lỗi API] {e}")
        return []

def analyze_match_with_ai(home, away, favorite, spread, bookie):
    """
    Hàm này yêu cầu AI đóng vai chuyên gia soi kèo
    """
    system_prompt = """
    You are a professional NBA Betting Analyst. 
    Your job is to analyze matchups based on team form, roster strength (2024-2025 season), and the betting line.
    """
    
    user_prompt = f"""
    Analyze this NBA match:
    - Matchup: {home} (Home) vs {away} (Away)
    - Current Line: {favorite} is favored by -{spread} points at {bookie}.
    
    Task 1: General Analysis
    - Who has better recent form? 
    - Is the spread too high or too low for these teams?
    - Pick the best bet (Spread or Moneyline).
    
    Task 2: Middle Betting Check
    - If the spread is large (>12), does the favorite typically start slow or allow garbage-time comebacks?
    
    Output strictly in this format:
    PICK: [Team Name covering the spread OR Moneyline Winner]
    CONFIDENCE: [0-100]
    ANALYSIS: [One sentence explaining the form/matchup key factor]
    MIDDLE_POTENTIAL: [YES/NO] - [Short reason why]
    """
    
    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        
        # Parse kết quả thủ công để dễ xử lý
        result = {"pick": "N/A", "confidence": 0, "analysis": "N/A", "middle": "N/A"}
        for line in content.split('\n'):
            if "PICK:" in line: result["pick"] = line.split("PICK:")[1].strip()
            if "CONFIDENCE:" in line: 
                try: result["confidence"] = int(''.join(filter(str.isdigit, line)))
                except: result["confidence"] = 50
            if "ANALYSIS:" in line: result["analysis"] = line.split("ANALYSIS:")[1].strip()
            if "MIDDLE_POTENTIAL:" in line: result["middle"] = line.split("MIDDLE_POTENTIAL:")[1].strip()
            
        return result
        
    except Exception as e:
        print(f"[Lỗi AI] {e}")
        return None

def main_program():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🏀 ĐANG KHỞI CHẠY HỆ THỐNG PHÂN TÍCH NBA...")
    events = get_nba_odds()
    
    if not events:
        print("❌ Không tìm thấy trận đấu nào trên sàn.")
        return

    print(f"👉 Tìm thấy {len(events)} trận đấu sắp/đang diễn ra.\n")

    for event in events:
        home = event['home_team']
        away = event['away_team']
        
        # Tìm kèo tốt nhất để phân tích
        best_spread = 0
        favorite = ""
        bookie_name = ""
        
        # Lấy dữ liệu Spread đầu tiên tìm thấy (đơn giản hóa)
        if not event['bookmakers']: continue
        
        # Logic tìm kèo spread lớn nhất (để đánh giá Middle)
        for bookie in event['bookmakers']:
            for market in bookie['markets']:
                if market['key'] == 'spreads':
                    for outcome in market['outcomes']:
                        spread_val = abs(outcome['point'])
                        if spread_val > best_spread:
                            best_spread = spread_val
                            bookie_name = bookie['title']
                            if outcome['point'] < 0:
                                favorite = outcome['name']
                            else:
                                favorite = away if outcome['name'] == home else home
        
        if best_spread == 0: continue # Không có kèo spread

        # === BẮT ĐẦU PHÂN TÍCH ===
        print(f"🔹 {home} vs {away} | Kèo: {favorite} -{best_spread}")
        print(f"   ... Đang gửi dữ liệu cho AI phân tích ...")
        
        ai_data = analyze_match_with_ai(home, away, favorite, best_spread, bookie_name)
        
        if ai_data:
            # 1. Hiển thị phân tích chính (General Betting)
            print(f"   🧠 AI GỢI Ý (PICK): \033[1m{ai_data['pick']}\033[0m")
            print(f"   📊 Độ tin cậy: {ai_data['confidence']}/100")
            print(f"   📝 Lý do: {ai_data['analysis']}")
            
            # 2. Logic phụ: Kiểm tra Middle Betting
            # Chỉ hiện nếu Kèo sâu VÀ AI nhận định có tiềm năng (YES)
            if best_spread >= MIDDLE_BET_THRESHOLD:
                is_middle_good = "YES" in ai_data['middle'].upper()
                
                if is_middle_good:
                    print(f"\n   🔥🔥 \033[93mCƠ HỘI MIDDLE BETTING!\033[0m 🔥🔥")
                    print(f"   Strategy: Pre-bet {away if favorite == home else home} (+{best_spread})")
                    print(f"   Lý do AI: {ai_data['middle']}")
                else:
                    print(f"   ⚠️ Middle Betting: Không khuyến nghị (Dù kèo sâu nhưng AI đánh giá rủi ro).")
            
        print("-" * 60)
        time.sleep(2) # Delay nhỏ để tránh spam API quá nhanh

if __name__ == "__main__":
    main_program()