import sqlite3
import pandas as pd
import numpy as np
import time
import re
import io
import json
import logging
import random
import platform
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager

# --- SELENIUM IMPORTS ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

# --- DATABASE FUNCTIONS ---
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

def migrate_db(conn):
    c = conn.cursor()
    c.execute("PRAGMA table_info(market_logs)")
    columns = [info[1] for info in c.fetchall()]
    
    new_columns = {
        'm1_call_iv': 'REAL',
        'm1_put_iv': 'REAL',
        'skew_put_avg_9_10_11_iv': 'REAL',
        'skew_call_avg_9_10_11_iv': 'REAL',
        'm1_month': 'TEXT',
        'skew_index': 'REAL',
        'm1_iv': 'REAL'
    }
    
    for col_name, col_type in new_columns.items():
        if col_name not in columns:
            logger.info(f"🛠️ Migrating DB: Adding column '{col_name}'...")
            try: c.execute(f"ALTER TABLE market_logs ADD COLUMN {col_name} {col_type}")
            except Exception as e: logger.error(f"Migration failed for {col_name}: {e}")

def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS market_logs (
                timestamp DATETIME PRIMARY KEY,
                spot_price REAL NOT NULL,
                m1_straddle REAL,
                m1_call_iv REAL,
                m1_put_iv REAL,
                m1_iv REAL,
                m2_iv REAL,
                m3_iv REAL,
                skew_put_avg_9_10_11_iv REAL,
                skew_call_avg_9_10_11_iv REAL,
                skew_index REAL,
                india_vix REAL,
                m1_month TEXT
            )
        ''')
        migrate_db(conn)
    logger.info(f"✅ Database {DB_NAME} checked/ready.")

# --- UTILS ---
def validate_spot_price(price: float) -> bool:
    if not isinstance(price, (int, float)): return False
    return SPOT_PRICE_MIN <= price <= SPOT_PRICE_MAX

def validate_vix(vix: float) -> bool:
    return isinstance(vix, (int, float)) and VIX_MIN <= vix <= VIX_MAX

# --- SCRAPERS ---
def check_scrape_permission(driver: webdriver.Chrome) -> bool:
    logger.info("   -> Checking Scrape Permission (Visual & Date)...")
    try:
        driver.get(HOME_URL)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        open_indicators = ["Normal Market is Open", "Market Status : Open", "Capital Market : Open", "NIFTY 50 : Open"]
        for indicator in open_indicators:
            if re.search(indicator, page_text, re.IGNORECASE):
                logger.info(f"   ✅ Detected Status: '{indicator}' -> GO")
                return True

        date_pattern = r"(\d{2}-[A-Za-z]{3}-\d{4})"
        matches = re.findall(date_pattern, page_text)
        today_str = datetime.now().strftime("%d-%b-%Y")
        
        if matches:
            for date_str in matches:
                if date_str.lower() == today_str.lower():
                    logger.info(f"   ✅ Detected Today's Date ({date_str}) in header -> GO")
                    return True
        
        today_num = datetime.now().strftime("%d-%m-%Y")
        if today_num in page_text:
             logger.info(f"   ✅ Detected Today's Date ({today_num}) -> GO")
             return True

        if "Normal Market has Closed" in page_text:
            logger.warning(f"   ⛔ Market Closed. Checking for data anyway...")
            return True 

        logger.warning("   ⚠️ No Open status OR Today's date found.")
        return True 
    except Exception as e:
        logger.error(f"   ❌ Scrape Check Failed: {e}")
        return True 

def check_live_holiday(driver: webdriver.Chrome) -> bool:
    try:
        driver.get(HOLIDAY_API_URL)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        json_str = driver.find_element(By.TAG_NAME, "body").text
        data = json.loads(json_str)
        today_str = datetime.now().strftime("%d-%b-%Y")
        for h in data.get('FO', []):
            if h.get('tradingDate') == today_str:
                logger.info(f"🚫 Today is a Trading Holiday: {h.get('description')}")
                return True
        return False
    except: return False

def get_india_vix_nse(driver: webdriver.Chrome) -> float:
    try:
        driver.get(VIX_API_URL)
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        data = json.loads(driver.find_element(By.TAG_NAME, "body").text)
        for item in data.get('data', []):
            if (item.get('index') or item.get('indexSymbol')) == "INDIA VIX":
                vix = float(item.get('last', 0))
                if validate_vix(vix): return round(vix, 2)
    except: pass
    try:
        driver.get(INDICES_URL)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        match = re.search(r'INDIA\s*VIX.*?(\d{2}\.\d{2})', driver.find_element(By.TAG_NAME, "body").text)
        if match: return round(float(match.group(1)), 2)
    except: pass
    return 0.00

# --- SMART DROPDOWN FINDER ---
def find_expiry_dropdown(driver: webdriver.Chrome):
    """
    Locates the dropdown using Label proximity (Robust) and ID (Fast).
    """
    try:
        # 1. Try Specific ID first
        try:
            return driver.find_element(By.ID, "select_opt_expiry_date")
        except: pass

        # 2. LABEL STRATEGY (Most Robust based on Screenshot)
        try:
            dropdown = driver.find_element(By.XPATH, "//label[contains(text(),'Expiry Date')]/following::select[1]")
            logger.info("   🔎 Found Expiry Dropdown via Label Match.")
            return dropdown
        except: pass

        # 3. Content Scan Fallback
        selects = driver.find_elements(By.TAG_NAME, "select")
        for sel in selects:
            try:
                if re.search(r"\d{2}-[A-Za-z]{3}-\d{4}", sel.text):
                    return sel
            except StaleElementReferenceException:
                continue
        
        return None
    except Exception as e:
        logger.error(f"   ❌ Dropdown search error: {e}")
        return None

def get_monthly_expiries(driver: webdriver.Chrome, dropdown_element) -> Optional[List[Tuple]]:
    try:
        options = dropdown_element.find_elements(By.TAG_NAME, "option")
        date_map = []
        for opt in options:
            val = opt.get_attribute("value")
            text = opt.text
            if "Select" in text or not val: continue
            try: date_map.append((datetime.strptime(text, "%d-%b-%Y"), val))
            except: continue
        
        if not date_map: return None
        date_map.sort(key=lambda x: x[0])
        month_groups = {}
        for d_obj, val in date_map:
            month_groups.setdefault((d_obj.year, d_obj.month), []).append((d_obj, val))
        
        return [month_groups[k][-1] for k in sorted(month_groups.keys())][:3]
    except: return None

def switch_expiry(driver: webdriver.Chrome, dropdown, value: str) -> bool:
    try:
        # Trigger change event via JS
        driver.execute_script("arguments[0].value = arguments[1];", dropdown, value)
        driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", dropdown)
        
        # Wait for loading overlay to disappear if present
        try:
            WebDriverWait(driver, 2).until(EC.invisibility_of_element_located((By.CLASS_NAME, "loader")))
        except: pass
        
        time.sleep(6) # Wait for table AJAX
        return True
    except: return False

def extract_page_timestamp(driver: webdriver.Chrome) -> str:
    try:
        src = driver.page_source
        match = re.search(r"As on\s+([A-Za-z]{3}\s+\d{2},\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+IST)", src)
        if match: return match.group(1)
        return "Unknown"
    except: return "Error"

def scrape_table_data(driver: webdriver.Chrome) -> Optional[Dict]:
    try:
        try: WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "optionChainTable-indices")))
        except: pass

        # --- DATA FRESHNESS CHECK ---
        page_time_str = extract_page_timestamp(driver)
        logger.info(f"       🕒 Data Timestamp: {page_time_str}")

        spot = 0.0
        
        # --- ROBUST SPOT PRICE EXTRACTION ---
        try:
            element = driver.find_element(By.ID, "equity_underlyingVal")
            text = element.text.strip() 
            match = re.search(r'([0-9,]+\.\d{2})', text)
            if match:
                spot = float(match.group(1).replace(",", ""))
                logger.info(f"       ✅ Spot found via ID: {spot}")
        except: 
            pass

        # Method 2: Fallback Regex on Page Source (Context Aware)
        if spot == 0.0:
            src = driver.page_source
            match = re.search(r'NIFTY\s*(?:50)?\s*[:\s]\s*([0-9,]+\.\d{2})', src, re.IGNORECASE)
            if match:
                spot = float(match.group(1).replace(",", ""))
                logger.info(f"       ✅ Spot found via Regex: {spot}")

        # --- VALIDATION ---
        if not validate_spot_price(spot):
            logger.error(f"❌ Invalid Spot Price Scraped: {spot}")
            raise Exception("Spot Price Validation Failed")

        # --- TABLE PARSING ---
        try:
            dfs = pd.read_html(io.StringIO(driver.page_source))
        except ValueError:
             raise Exception("No tables found in page source")
             
        if not dfs: raise Exception("No tables parsed")
        
        df = next((d for d in dfs if d.shape[0] > 10 and d.shape[1] > 10), None)
        if df is None: raise Exception("Chain table not found")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
        
        strike_col = next((c for c in df.columns if "STRIKE" in str(c).upper()), None)
        iv_cols = [c for c in df.columns if "IV" in str(c).upper()]
        ltp_cols = [c for c in df.columns if "LTP" in str(c).upper()]
        
        if not strike_col or len(iv_cols) < 2: raise Exception("Missing columns")

        df = df[pd.to_numeric(df[strike_col], errors='coerce').notnull()].copy()
        df[strike_col] = df[strike_col].astype(float)
        
        # --- 1. GENERAL CALCULATIONS (ATM / Straddle) ---
        df['diff'] = abs(df[strike_col] - spot)
        df_sorted = df.sort_values('diff').reset_index(drop=True)
        
        atm = df_sorted.iloc[0]
        def safe(v):
            try: return float(v) if str(v).strip() != '-' else 0.0
            except: return 0.0

        straddle = safe(atm[ltp_cols[0]]) + safe(atm[ltp_cols[1]])
        
        atm_call_iv = 0.0
        atm_put_iv = 0.0
        for i in range(min(5, len(df_sorted))):
            c_iv = safe(df_sorted.iloc[i][iv_cols[0]])
            p_iv = safe(df_sorted.iloc[i][iv_cols[1]])
            if c_iv > 0 and p_iv > 0: 
                atm_call_iv = c_iv
                atm_put_iv = p_iv
                break
        
        avg_iv = (atm_call_iv + atm_put_iv) / 2 if (atm_call_iv > 0 and atm_put_iv > 0) else 0.0
        
        # --- 2. REFINED SKEW: PRICE-BASED BASKET SELECTION ---
        skew_val = 0.0
        avg_put_iv_basket = 0.0
        avg_call_iv_basket = 0.0
        
        try:
            df_100 = df[df[strike_col] % 100 == 0].copy()
            target_put_center = spot * 0.98
            target_call_center = spot * 1.02
            basket_offsets = [0, -100, 100]
            p_targets = [target_put_center + x for x in basket_offsets]
            c_targets = [target_call_center + x for x in basket_offsets]
            
            def get_basket_avg(targets, iv_col_name):
                valid_ivs = []
                for tgt in targets:
                    if not df_100.empty:
                        closest_idx = (df_100[strike_col] - tgt).abs().idxmin()
                        iv = safe(df_100.loc[closest_idx][iv_col_name])
                        if iv > 0: valid_ivs.append(iv)
                if valid_ivs:
                    return sum(valid_ivs) / len(valid_ivs)
                return 0.0

            avg_put_iv_basket = get_basket_avg(p_targets, iv_cols[1]) 
            avg_call_iv_basket = get_basket_avg(c_targets, iv_cols[0]) 

            if avg_put_iv_basket > 0 and avg_call_iv_basket > 0:
                skew_val = avg_put_iv_basket - avg_call_iv_basket
                    
        except Exception as e:
            logger.warning(f"Skew calc warning: {e}")

        return {
            "spot": round(spot, 2), 
            "straddle": round(straddle, 2), 
            "m1_call_iv": round(atm_call_iv, 2),
            "m1_put_iv": round(atm_put_iv, 2),
            "m1_iv": round(avg_iv, 2),
            "skew_index": round(skew_val, 2), 
            "skew_put_avg_9_10_11_iv": round(avg_put_iv_basket, 2),
            "skew_call_avg_9_10_11_iv": round(avg_call_iv_basket, 2)
        }
    except Exception as e:
        logger.warning(f"Parse error: {e}")
        return None

# --- MAIN EXECUTION ---
def update_market_data():
    logger.info("🔄 STARTING UPDATE PROCESS")
    
    # 1. DETECT OPERATING SYSTEM
    current_os = platform.system()
    logger.info(f"🖥️ Detected OS: {current_os}")

    options = Options()
    
    # 2. APPLY OS-SPECIFIC FLAGS
    if current_os == "Linux":
        logger.info("   -> Applying Linux/GitHub Actions optimizations...")
        options.add_argument("--headless=new") # <--- HEADLESS MODE ENABLED
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage") 
        options.add_argument("--no-zygote")
        options.add_argument("--single-process")
        options.set_capability("pageLoadStrategy", "eager")
        
    else:
        logger.info("   -> Applying Windows/Local optimizations...")
        options.add_argument("--headless=new") # <--- HEADLESS MODE ENABLED FOR WINDOWS TOO
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.set_capability("pageLoadStrategy", "normal")

    # Common Settings
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument("--ignore-certificate-errors")

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
    chosen_ua = random.choice(user_agents)
    options.add_argument(f"user-agent={chosen_ua}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": chosen_ua})
    
    driver.set_page_load_timeout(120) 
    driver.set_script_timeout(120)
    
    try:
        can_scrape = check_scrape_permission(driver)
        
        if not can_scrape:
             if check_live_holiday(driver):
                logger.info("🚫 Market is CLOSED (Holiday). Stopping.")
                driver.quit()
                return

        logger.info("   -> Initializing Session...")
        driver.get(HOME_URL)
        time.sleep(5)
        
        logger.info("   -> Loading Option Chain...")
        driver.get(OPTION_CHAIN_URL)
        
        # --- ROBUST WAIT (90s) ---
        try:
            WebDriverWait(driver, 90).until(EC.presence_of_element_located((By.ID, "optionChainTable-indices")))
        except TimeoutException:
            logger.error("❌ TIMEOUT: Option chain table did not load.")
            driver.execute_script("window.scrollBy(0, 500)")
            return

        time.sleep(5) 

        dropdown = find_expiry_dropdown(driver)
        if not dropdown: 
            logger.error("❌ Expiry Dropdown not found (FATAL).")
            return

        expiries = get_monthly_expiries(driver, dropdown)
        if not expiries: 
            logger.error("❌ No expiries found.")
            return

        live_data = {}
        m1_month_str = ""

        for i, (date_obj, val) in enumerate(expiries):
            label = f"m{i+1}"
            logger.info(f"   -> Scraping {label.upper()} ({date_obj.strftime('%d-%b')})...")
            
            if label == "m1":
                m1_month_str = date_obj.strftime("%b")

            dropdown = find_expiry_dropdown(driver)
            if dropdown:
                switch_expiry(driver, dropdown, val)
                row_data = scrape_table_data(driver)
                
                if row_data:
                    if label == "m1":
                        live_data.update({
                            'spot_price': row_data['spot'],
                            'm1_straddle': row_data['straddle'],
                            'm1_call_iv': row_data['m1_call_iv'],
                            'm1_put_iv': row_data['m1_put_iv'],
                            'm1_iv': row_data['m1_iv'],
                            'skew_index': row_data['skew_index'],
                            'skew_put_avg_9_10_11_iv': row_data['skew_put_avg_9_10_11_iv'],
                            'skew_call_avg_9_10_11_iv': row_data['skew_call_avg_9_10_11_iv']
                        })
                    elif label == "m2": live_data['m2_iv'] = row_data['m1_iv'] 
                    elif label == "m3": live_data['m3_iv'] = row_data['m1_iv']
            else:
                logger.error("❌ Dropdown lost during iteration.")

        live_data['india_vix'] = get_india_vix_nse(driver)
        live_data['m1_month'] = m1_month_str

        if live_data.get('spot_price', 0) == 0:
            logger.error("❌ Scrape Failed: Spot Price is 0.")
            return

        # Backfill
        if live_data.get('m3_iv', 0) == 0:
            if live_data.get('m2_iv', 0) > 0: live_data['m3_iv'] = live_data['m2_iv']
            elif live_data.get('m1_iv', 0) > 0: live_data['m3_iv'] = live_data['m1_iv']

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        schema_keys = [
            'spot_price', 'm1_straddle', 'm1_call_iv', 'm1_put_iv', 'm1_iv', 
            'm2_iv', 'm3_iv', 'skew_put_avg_9_10_11_iv', 'skew_call_avg_9_10_11_iv', 
            'skew_index', 'india_vix', 'm1_month'
        ]
        
        for k in schema_keys: 
            if k not in live_data: 
                live_data[k] = 0.00 if k != 'm1_month' else ""
            elif k != 'm1_month':
                live_data[k] = round(live_data[k], 2)

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        today_date = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute("DELETE FROM market_logs WHERE timestamp LIKE ?", (f"{today_date}%",))
        cursor.execute('''
            INSERT OR REPLACE INTO market_logs 
            (timestamp, spot_price, m1_straddle, m1_call_iv, m1_put_iv, m1_iv, m2_iv, m3_iv, 
             skew_put_avg_9_10_11_iv, skew_call_avg_9_10_11_iv, skew_index, india_vix, m1_month)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp_str, live_data['spot_price'], live_data['m1_straddle'], 
            live_data['m1_call_iv'], live_data['m1_put_iv'], live_data['m1_iv'],
            live_data['m2_iv'], live_data['m3_iv'], 
            live_data['skew_put_avg_9_10_11_iv'], live_data['skew_call_avg_9_10_11_iv'],
            live_data['skew_index'], live_data['india_vix'], live_data['m1_month']
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"✅ DATA SAVED for {today_date}!")
        logger.info(f"   Spot: {live_data['spot_price']} | M1 Call IV: {live_data['m1_call_iv']:.2f}% | Skew Index: {live_data['skew_index']:.2f}")

    except Exception as e:
        logger.error(f"❌ Main Loop Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    init_db()
    update_market_data()
