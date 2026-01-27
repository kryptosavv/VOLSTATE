import sqlite3
import pandas as pd
import numpy as np
import time
import re
import io
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
VIX_API_URL = "https://www.nseindia.com/api/allIndices"
HOME_URL = "https://www.nseindia.com/"

# Validation ranges
SPOT_PRICE_MIN = 15000
SPOT_PRICE_MAX = 50000 
VIX_MIN = 5.0
VIX_MAX = 100.0

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # 1. DROP OLD TABLES (To apply new schema)
        c.execute("DROP TABLE IF EXISTS market_logs")
        c.execute("DROP TABLE IF EXISTS market_logs_backup")
        
        # 2. CREATE NEW SIMPLIFIED TABLE
        c.execute('''
            CREATE TABLE market_logs (
                timestamp DATETIME PRIMARY KEY,
                spot_price REAL NOT NULL,
                m1_straddle REAL,
                m1_iv REAL,
                m2_iv REAL,
                m3_iv REAL,
                m1_dte INTEGER,
                skew_index REAL,
                india_vix REAL
            )
        ''')
    logger.info(f"✅ Database {DB_NAME} ready with new schema.")

def validate_spot_price(price: float) -> bool:
    if not isinstance(price, (int, float)): return False
    return SPOT_PRICE_MIN <= price <= SPOT_PRICE_MAX

def validate_vix(vix: float) -> bool:
    if not isinstance(vix, (int, float)): return False
    return VIX_MIN <= vix <= VIX_MAX

def is_weekend() -> bool:
    today = datetime.now()
    if today.weekday() >= 5:
        logger.info(f"🚫 Today is {today.strftime('%A')} (Weekend). No scraping.")
        return True
    return False

def check_live_holiday(driver: webdriver.Chrome) -> bool:
    try:
        logger.info("   -> Checking NSE Holiday API...")
        driver.get(HOLIDAY_API_URL)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        json_str = driver.find_element(By.TAG_NAME, "body").text
        data = json.loads(json_str)
        holidays = data.get('FO', [])
        today_str = datetime.now().strftime("%d-%b-%Y")
        for h in holidays:
            if h.get('tradingDate') == today_str:
                logger.info(f"🚫 Today is a Trading Holiday: {h.get('description')}")
                return True
        return False
    except Exception as e:
        logger.error(f"⚠️ Holiday Check Failed (Assuming Open): {e}")
        return False

# --- NEW API-BASED VIX LOGIC ---
def get_india_vix_nse(driver: webdriver.Chrome) -> float:
    # METHOD 1: Direct API (Best & Fastest)
    # This visits the same JSON API the browser uses to populate the table
    try:
        logger.info("   -> Fetching VIX from API (api/allIndices)...")
        driver.get(VIX_API_URL)
        
        # Wait for JSON text to appear in body
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        json_str = driver.find_element(By.TAG_NAME, "body").text
        
        data = json.loads(json_str)
        # Search for the VIX object in the list
        for item in data.get('data', []):
            name = item.get('index', '') or item.get('indexSymbol', '')
            if name == "INDIA VIX":
                vix = float(item.get('last', 0))
                if validate_vix(vix): return vix
    except Exception as e:
        logger.warning(f"      ⚠️ API VIX failed, trying fallback... ({e})")

    # METHOD 2: HTML Scraping (Indices Page Fallback)
    try:
        logger.info("   -> Fetching VIX from Indices Page (HTML Fallback)...")
        driver.get(INDICES_URL)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # We use .text instead of .page_source as it handles spacing better
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        # Look for "INDIA VIX" followed by numbers
        match = re.search(r'INDIA\s*VIX.*?(\d{2}\.\d{2})', page_text)
        if match:
            vix = float(match.group(1))
            if validate_vix(vix): return vix
    except: pass
    
    logger.warning("❌ Could not fetch VIX from API or HTML.")
    return 0.0

def find_expiry_dropdown(driver: webdriver.Chrome):
    try:
        current_year = str(datetime.now().year)
        next_year = str(datetime.now().year + 1)
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "select")))
        selects = driver.find_elements(By.TAG_NAME, "select")
        for sel in selects:
            if current_year in sel.text or next_year in sel.text:
                return sel
        return None
    except: return None

def get_monthly_expiries(driver: webdriver.Chrome, dropdown_element) -> Optional[List[Tuple]]:
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

def switch_expiry(driver: webdriver.Chrome, dropdown_element, value_to_select: str) -> bool:
    try:
        old_content = driver.page_source
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
        
        for _ in range(10):
            time.sleep(1)
            if driver.page_source != old_content: return True
        return False
    except: return False

def scrape_table_data(driver: webdriver.Chrome) -> Optional[Dict]:
    try:
        page_source = driver.page_source
        match_spot = re.search(r'NIFTY\s+([0-9,]+\.\d{2})', page_source)
        if not match_spot: 
            match_spot = re.search(r'([0-9]{5}\.\d{2})', page_source)
            if not match_spot: raise Exception("Spot not found")
        
        spot_price = float(match_spot.group(1).replace(",", ""))
        
        if not validate_spot_price(spot_price): 
            logger.warning(f"Spot price {spot_price} looks weird, but continuing.")

        html_buffer = io.StringIO(page_source)
        dfs = pd.read_html(html_buffer)
        if not dfs: raise Exception("No tables found")
        
        chain_df = None
        for df in dfs:
            if df.shape[0] > 10 and df.shape[1] > 10:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
                col_str = ' '.join(df.columns.astype(str).str.upper())
                if 'STRIKE' in col_str and col_str.count('IV') >= 2:
                    chain_df = df
                    break
        
        if chain_df is None: raise Exception("Option chain table not found")

        strike_col = next((c for c in chain_df.columns if "STRIKE" in str(c).upper()), None)
        iv_cols = [c for c in chain_df.columns if "IV" in str(c).upper()]
        ltp_cols = [c for c in chain_df.columns if "LTP" in str(c).upper()]
        
        if not strike_col or len(iv_cols) < 2: raise Exception("Missing required columns")

        chain_df = chain_df[pd.to_numeric(chain_df[strike_col], errors='coerce').notnull()].copy()
        chain_df[strike_col] = chain_df[strike_col].astype(float)
        chain_df = chain_df.sort_values(strike_col).reset_index(drop=True)
        
        chain_df['diff'] = abs(chain_df[strike_col] - spot_price)
        atm_idx = chain_df['diff'].idxmin()
        atm_row = chain_df.loc[atm_idx]
        atm_strike = atm_row[strike_col]

        def safe_float(val):
            try: return float(val) if val != '-' else 0.0
            except: return 0.0

        call_iv = safe_float(atm_row[iv_cols[0]])
        put_iv = safe_float(atm_row[iv_cols[1]])
        avg_iv = (call_iv + put_iv) / 2 if (call_iv and put_iv) else (call_iv or put_iv)
        straddle_price = safe_float(atm_row[ltp_cols[0]]) + safe_float(atm_row[ltp_cols[1]])

        skew_index = 0.0
        try:
            unique_strikes = sorted(chain_df[strike_col].unique())
            if len(unique_strikes) > 1:
                step = unique_strikes[1] - unique_strikes[0]
                target_put = atm_strike - (step * 5)
                target_call = atm_strike + (step * 5)
                
                put_row = chain_df.iloc[(chain_df[strike_col] - target_put).abs().argsort()[:1]]
                call_row = chain_df.iloc[(chain_df[strike_col] - target_call).abs().argsort()[:1]]
                
                if not put_row.empty and not call_row.empty:
                    p_iv = safe_float(put_row[iv_cols[1]].values[0])
                    c_iv = safe_float(call_row[iv_cols[0]].values[0])
                    if p_iv > 0 and c_iv > 0: skew_index = p_iv - c_iv
        except: pass

        return { "spot": spot_price, "iv": avg_iv, "straddle": straddle_price, "skew": skew_index }
    except Exception as e:
        logger.warning(f"Parse error: {e}")
        return None

def get_live_data_diagnostics() -> Optional[Dict]:
    if is_weekend(): return None

    options = Options()
    options.add_argument("--headless=new") 
    options.add_argument("--disable-gpu") 
    options.add_argument("--no-sandbox")   
    options.add_argument("--disable-dev-shm-usage") 
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # --- STEALTH MODE ENABLED ---
    options.add_argument("--disable-blink-features=AutomationControlled") 
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    logger.info("🚀 Launching Headless Browser (Stealth Mode)...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    final_record = {}

    try:
        logger.info("   -> Initializing Session...")
        driver.get(HOME_URL)
        time.sleep(3)

        if check_live_holiday(driver):
            return None

        logger.info("   -> Loading Option Chain...")
        driver.get(OPTION_CHAIN_URL)
        time.sleep(8) 
        
        dropdown = find_expiry_dropdown(driver)
        if not dropdown: 
            logger.error("❌ Expiry Dropdown not found.")
            return None

        expiries = get_monthly_expiries(driver, dropdown)
        if not expiries: 
            logger.error("❌ No expiries found.")
            return None

        for i, (date_obj, val) in enumerate(expiries):
            label = f"m{i+1}"
            logger.info(f"   -> Scraping {label.upper()} ({date_obj.strftime('%d-%b')})...")
            
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
                logger.warning(f"       ⚠️ Failed to scrape {label}")

        final_record['india_vix'] = get_india_vix_nse(driver)
        return final_record

    except Exception as e:
        logger.error(f"❌ Critical Error: {e}")
        return None
    finally:
        try: driver.quit()
        except: pass

def update_market_data():
    logger.info("🔄 STARTING UPDATE PROCESS")
    live_data = get_live_data_diagnostics()
    
    if not live_data:
        logger.info("⏹️ No data collected (Holiday/Weekend/Error).")
        return
    
    if live_data.get('spot_price', 0) == 0:
        logger.error("❌ Scrape Failed: Spot Price is 0.")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    defaults = {'spot_price': 0, 'm1_straddle': 0, 'm1_iv': 0, 'm2_iv': 0, 'm3_iv': 0, 'm1_dte': 0, 'skew_index': 0, 'india_vix': 0}
    for k, v in defaults.items():
        if k not in live_data: live_data[k] = v

    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # One Record Per Day Logic
    cursor.execute("DELETE FROM market_logs WHERE timestamp LIKE ?", (f"{today_date}%",))
    
    cursor.execute('''
        INSERT OR REPLACE INTO market_logs 
        (timestamp, spot_price, m1_straddle, m1_iv, m2_iv, m3_iv, m1_dte, skew_index, india_vix)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp_str, live_data['spot_price'], live_data['m1_straddle'], live_data['m1_iv'],
        live_data['m2_iv'], live_data['m3_iv'], live_data['m1_dte'], live_data['skew_index'], live_data['india_vix']
    ))
    conn.commit()
    conn.close()
    
    logger.info(f"✅ DATA SAVED for {today_date}!")
    logger.info(f"   Spot: {live_data['spot_price']} | M1 IV: {live_data['m1_iv']:.2f}% | VIX: {live_data['india_vix']}")
    
    # 

if __name__ == "__main__":
    init_db()
    update_market_data()
