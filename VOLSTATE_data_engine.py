import sqlite3
import pandas as pd
import numpy as np
import time
import re
import io
import json
import logging
from datetime import datetime

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- SELENIUM ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIG ---
DB_NAME = "market_data.db"
OPTION_CHAIN_URL = "https://www.nseindia.com/option-chain"
INDICES_URL = "https://www.nseindia.com/market-data/live-market-indices"
HOLIDAY_API_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
HOME_URL = "https://www.nseindia.com/"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS market_logs (
            timestamp DATETIME PRIMARY KEY,
            spot_price REAL,
            m1_straddle REAL,
            m1_iv REAL,
            m2_iv REAL,
            m3_iv REAL,
            m1_dte INTEGER,
            skew_index REAL,
            vvix REAL
        )
    ''')
    conn.commit()
    conn.close()
    logging.info(f"✅ Database {DB_NAME} ready.")

def is_weekend():
    today = datetime.now()
    if today.weekday() >= 5:
        logging.info(f"🚫 Today is {today.strftime('%A')} (Weekend). No scraping.")
        return True
    return False

def check_live_holiday(driver):
    try:
        logging.info("   -> Checking NSE Holiday API...")
        driver.get(HOLIDAY_API_URL)
        json_str = driver.find_element(By.TAG_NAME, "body").text
        data = json.loads(json_str)
        holidays = data.get('FO', [])
        today_str = datetime.now().strftime("%d-%b-%Y")
        
        for h in holidays:
            if h.get('tradingDate') == today_str:
                logging.info(f"🚫 Today is a Trading Holiday: {h.get('description')}")
                return True
        return False
    except Exception as e:
        logging.warning(f"⚠️ Holiday Check Failed (Assuming Open): {e}")
        return False

def extract_vix_from_text(page_text):
    match = re.search(r"INDIA VIX\s+([\d\.]+)", page_text)
    if match: return float(match.group(1))
    return 0.0

def get_india_vix_nse(driver):
    try:
        logging.info("   -> Attempting VIX from Indices Page...")
        driver.get(INDICES_URL)
        try: WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "liveIndexTable")))
        except: pass 
        page_text = driver.execute_script("return document.body.innerText;")
        val = extract_vix_from_text(page_text)
        if val > 0: return val
    except: pass

    try:
        logging.info("   -> Fallback: Attempting VIX from Home Page...")
        driver.get(HOME_URL)
        time.sleep(3)
        page_text = driver.execute_script("return document.body.innerText;")
        val = extract_vix_from_text(page_text)
        if val > 0: return val
    except: pass
    return 0.0

def find_expiry_dropdown(driver):
    try:
        current_year = str(datetime.now().year)
        next_year = str(datetime.now().year + 1)
        selects = driver.find_elements(By.TAG_NAME, "select")
        for sel in selects:
            if current_year in sel.text or next_year in sel.text:
                return sel
        return None
    except: return None

def get_monthly_expiries(driver, dropdown_element):
    try:
        options = dropdown_element.find_elements(By.TAG_NAME, "option")
        date_map = [] 
        for opt in options:
            val = opt.get_attribute("value")
            text = opt.text
            if "Select" in text or not val: continue
            try:
                d_obj = datetime.strptime(text, "%d-%b-%Y")
                date_map.append((d_obj, val))
            except: continue
        if not date_map: return None
        date_map.sort(key=lambda x: x[0])
        month_groups = {}
        for d_obj, val in date_map:
            key = (d_obj.year, d_obj.month)
            if key not in month_groups: month_groups[key] = []
            month_groups[key].append((d_obj, val))
        monthly_expiries = []
        for key in sorted(month_groups.keys()):
            monthly_expiries.append(month_groups[key][-1]) 
        return monthly_expiries[:3] 
    except: return None

def switch_expiry(driver, dropdown_element, value_to_select):
    try:
        driver.execute_script("""
            var select = arguments[0];
            var value = arguments[1];
            var options = Array.from(select.options);
            var optionToSelect = options.find(o => o.value === value);
            if (optionToSelect) {
                var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value").set;
                nativeInputValueSetter.call(select, value);
                select.dispatchEvent(new Event('change', { bubbles: true }));
                select.dispatchEvent(new Event('input', { bubbles: true }));
            }
        """, dropdown_element, value_to_select)
        time.sleep(5) 
        return True
    except: return False

def scrape_table_data(driver):
    try:
        page_source = driver.page_source
        match_spot = re.search(r'NIFTY\s+([0-9,]+\.\d{2})', page_source)
        if match_spot: spot_price = float(match_spot.group(1).replace(",", ""))
        else: return None

        html_buffer = io.StringIO(page_source)
        dfs = pd.read_html(html_buffer)
        if not dfs: return None
        chain_df = max(dfs, key=lambda df: df.shape[0] * df.shape[1])
        
        if isinstance(chain_df.columns, pd.MultiIndex):
            chain_df.columns = ['_'.join(map(str, col)).strip() for col in chain_df.columns.values]

        strike_col = next((c for c in chain_df.columns if "STRIKE" in str(c).upper()), None)
        iv_cols = [c for c in chain_df.columns if "IV" in str(c).upper()]
        ltp_cols = [c for c in chain_df.columns if "LTP" in str(c).upper()]

        if not strike_col or len(iv_cols) < 2 or len(ltp_cols) < 2: return None

        chain_df = chain_df[pd.to_numeric(chain_df[strike_col], errors='coerce').notnull()]
        chain_df[strike_col] = chain_df[strike_col].astype(float)
        chain_df = chain_df.sort_values(strike_col).reset_index(drop=True)
        
        chain_df['diff'] = abs(chain_df[strike_col] - spot_price)
        atm_idx = chain_df['diff'].idxmin()
        atm_row = chain_df.loc[atm_idx]
        atm_strike = atm_row[strike_col]
        
        call_iv = float(atm_row[iv_cols[0]]) if atm_row[iv_cols[0]] != '-' else 0
        put_iv = float(atm_row[iv_cols[1]]) if atm_row[iv_cols[1]] != '-' else 0
        avg_iv = (call_iv + put_iv) / 2 if (call_iv and put_iv) else (call_iv or put_iv)

        call_ltp = float(atm_row[ltp_cols[0]]) if atm_row[ltp_cols[0]] != '-' else 0
        put_ltp = float(atm_row[ltp_cols[1]]) if atm_row[ltp_cols[1]] != '-' else 0
        straddle_price = call_ltp + put_ltp

        skew_index = 0.0
        try:
            unique_strikes = sorted(chain_df[strike_col].unique())
            strike_step = unique_strikes[1] - unique_strikes[0] if len(unique_strikes) > 1 else 50
            otm_distance = strike_step * 5 
            target_put = atm_strike - otm_distance
            target_call = atm_strike + otm_distance
            
            put_row = chain_df.iloc[(chain_df[strike_col] - target_put).abs().argsort()[:1]]
            call_row = chain_df.iloc[(chain_df[strike_col] - target_call).abs().argsort()[:1]]
            
            if not put_row.empty and not call_row.empty:
                p_iv = float(put_row[iv_cols[1]].values[0]) if put_row[iv_cols[1]].values[0] != '-' else 0
                c_iv = float(call_row[iv_cols[0]].values[0]) if call_row[iv_cols[0]].values[0] != '-' else 0
                if p_iv > 0 and c_iv > 0: skew_index = p_iv - c_iv
        except: skew_index = 0.0

        return { "spot": spot_price, "iv": avg_iv, "straddle": straddle_price, "skew": skew_index }
    except: return None

def get_live_data_diagnostics():
    if is_weekend(): return None

    options = Options()
    options.add_argument("--headless=new") 
    options.add_argument("--disable-gpu") 
    options.add_argument("--no-sandbox")   
    options.add_argument("--disable-dev-shm-usage") 
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    logging.info("🚀 Launching Headless Browser...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    final_record = {}

    try:
        logging.info("   -> Initializing Session...")
        driver.get(HOME_URL)
        time.sleep(3)

        if check_live_holiday(driver):
            return None

        logging.info("   -> Loading Option Chain...")
        driver.get(OPTION_CHAIN_URL)
        time.sleep(8) 
        
        dropdown = find_expiry_dropdown(driver)
        if not dropdown: 
            logging.error("❌ Expiry Dropdown not found.")
            return None

        expiries = get_monthly_expiries(driver, dropdown)
        if not expiries: 
            logging.error("❌ No expiries found.")
            return None

        for i, (date_obj, val) in enumerate(expiries):
            label = f"m{i+1}"
            logging.info(f"   -> Scraping {label.upper()} ({date_obj.strftime('%d-%b')})...")
            dropdown = find_expiry_dropdown(driver)
            switch_expiry(driver, dropdown, val)
            row_data = scrape_table_data(driver)
            
            if row_data:
                if label == "m1":
                    final_record['spot_price'] = row_data['spot']
                    final_record['m1_straddle'] = row_data['straddle']
                    final_record['m1_iv'] = row_data['iv']
                    final_record['skew_index'] = row_data['skew']
                    dte = (date_obj - datetime.now()).days
                    final_record['m1_dte'] = max(0, dte)
                elif label == "m2": final_record['m2_iv'] = row_data['iv']
                elif label == "m3": final_record['m3_iv'] = row_data['iv']
            else:
                logging.warning(f"      ⚠️ Failed to scrape {label}")

        final_record['vvix'] = get_india_vix_nse(driver)
        return final_record

    except Exception as e:
        logging.error(f"❌ Critical Error: {e}")
        return None
    finally:
        try: driver.quit()
        except: pass

def update_market_data():
    logging.info("🔄 STARTING UPDATE PROCESS")
    live_data = get_live_data_diagnostics()
    
    if not live_data:
        logging.info("⏹️ No data collected (Holiday/Weekend/Error).")
        return
    
    if live_data.get('spot_price', 0) == 0:
        logging.error("❌ Scrape Failed: Spot Price is 0.")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    defaults = {'spot_price': 0, 'm1_straddle': 0, 'm1_iv': 0, 'm2_iv': 0, 'm3_iv': 0, 'm1_dte': 0, 'skew_index': 0, 'vvix': 0}
    for k, v in defaults.items():
        if k not in live_data: live_data[k] = v

    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # Single record per day logic
    cursor.execute("DELETE FROM market_logs WHERE timestamp LIKE ?", (f"{today_date}%",))
    
    cursor.execute('''
        INSERT OR REPLACE INTO market_logs 
        (timestamp, spot_price, m1_straddle, m1_iv, m2_iv, m3_iv, m1_dte, skew_index, vvix)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp_str, live_data['spot_price'], live_data['m1_straddle'], live_data['m1_iv'],
        live_data['m2_iv'], live_data['m3_iv'], live_data['m1_dte'], live_data['skew_index'], live_data['vvix']
    ))
    conn.commit()
    conn.close()
    
    logging.info(f"✅ DATA SAVED for {today_date}!")
    logging.info(f"   Spot: {live_data['spot_price']} | M1 IV: {live_data['m1_iv']:.2f}% | VIX: {live_data['vvix']}")

if __name__ == "__main__":
    init_db()
    update_market_data()