import sqlite3
import pandas as pd
import numpy as np
import time
import re
import io
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, List
from contextlib import contextmanager

# --- PLAYWRIGHT IMPORTS ---
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

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
        'm1_call_iv': 'REAL', 'm1_put_iv': 'REAL',
        'skew_put_avg_9_10_11_iv': 'REAL', 'skew_call_avg_9_10_11_iv': 'REAL',
        'm1_month': 'TEXT', 'm2_month': 'TEXT',
        'skew_index': 'REAL', 'm1_iv': 'REAL',
        'm2_straddle': 'REAL', 'cq_iv': 'REAL', 'nq_iv': 'REAL',
        'cq_straddle': 'REAL', 'nq_straddle': 'REAL',
        'cq_expiry': 'TEXT', 'nq_expiry': 'TEXT'
    }

    for col_name, col_type in new_columns.items():
        if col_name not in columns:
            try:
                c.execute(f"ALTER TABLE market_logs ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                pass

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
                m1_month TEXT,
                m2_month TEXT
            )
        ''')
        migrate_db(conn)
    logger.info(f"  ✅    Database {DB_NAME} checked/ready.")

# --- UTILS ---
def validate_spot_price(price: float) -> bool:
    if not isinstance(price, (int, float)): return False
    return SPOT_PRICE_MIN <= price <= SPOT_PRICE_MAX

def validate_vix(vix: float) -> bool:
    return isinstance(vix, (int, float)) and VIX_MIN <= vix <= VIX_MAX

# --- SCRAPERS ---

def stealth_setup(page):
    """Hide Playwright from NSE's bot detection."""
    page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

def check_scrape_permission(page) -> bool:
    logger.info("   -> Checking Scrape Permission...")
    try:
        page.goto(HOME_URL, wait_until="domcontentloaded")
        stealth_setup(page)
        
        try:
            page.wait_for_selector("body", timeout=30000)
        except PlaywrightTimeoutError:
             logger.warning("   ⚠️   Timeout waiting for body. Proceeding to check text anyway.")

        page_text = page.locator("body").inner_text()

        if "Market Status" not in page_text and "NIFTY 50" not in page_text:
            logger.warning("   ⚠️   Session check weak. Refreshing Home Page...")
            time.sleep(2)
            page.reload(wait_until="domcontentloaded")
            time.sleep(5)
            page_text = page.locator("body").inner_text()

        open_indicators = ["Normal Market is Open", "Market Status : Open", "Capital Market : Open", "NIFTY 50 : Open"]
        for indicator in open_indicators:
            if re.search(indicator, page_text, re.IGNORECASE):
                logger.info(f"     ✅    Detected Status: '{indicator}' -> GO")
                return True

        today_str = datetime.now().strftime("%d-%b-%Y")
        today_num = datetime.now().strftime("%d-%m-%Y")
        
        content = page.content()
        if today_str.lower() in content.lower() or today_num in content:
            logger.info(f"     ✅    Detected Today's Date -> GO")
            return True
            
        if "Normal Market has Closed" in page_text:
            logger.warning(f"     ⛔    Market Closed. Checking for data anyway...")
            return True

        return True
    except Exception as e:
        logger.error(f"     ❌    Scrape Check Failed: {e}")
        return True

def get_india_vix_nse(page) -> float:
    """
    ULTRA-ROBUST VIX SCRAPER (Ported from Selenium Logic in gee.docx)
    Uses XPath-like row detection on Indices page + Home Page Regex Fallback.
    """
    logger.info("   -> Fetching INDIA VIX...")

    # --- METHOD 1: INDICES PAGE TABLE ---
    try:
        page.goto(INDICES_URL, wait_until="domcontentloaded", timeout=45000)
        stealth_setup(page)
        
        # Wait for network idle (Critical for Table Load)
        try:
             page.wait_for_load_state("networkidle", timeout=10000)
        except: pass

        # Wait for ANY table row to ensure table exists
        try:
             page.wait_for_selector("tr", timeout=10000)
        except: pass

        # Find the row containing "INDIA VIX"
        # Selenium: driver.find_element(By.XPATH, "//tr[contains(., 'INDIA VIX')]")
        try:
            vix_row = page.locator("tr").filter(has_text="INDIA VIX").first
            if vix_row.is_visible():
                row_text = vix_row.inner_text()
                matches = re.findall(r"(\d{1,2}\.\d{2})", row_text)

                for m in matches:
                    val = float(m)
                    if validate_vix(val):
                        logger.info(f"     ✅    VIX Found (Indices Table): {val}")
                        return round(val, 2)
        except Exception as e:
            logger.warning(f"     ⚠️   Indices Table scan failed: {e}")

    except Exception as e:
        logger.error(f"     ❌    VIX Indices Page Error: {e}")

    # --- METHOD 2: HOME PAGE FALLBACK ---
    try:
        logger.info("   -> Trying Home Page Ticker (Fallback)...")
        page.goto(HOME_URL, wait_until="domcontentloaded", timeout=45000)
        stealth_setup(page)
        page.wait_for_selector("body", timeout=15000)
        
        # Simple scroll to trigger lazy loading
        try: page.evaluate("window.scrollBy(0, 100)")
        except: pass

        page_text = page.locator("body").inner_text()
        
        # Regex from gee.docx
        match = re.search(r'INDIA\s*VIX.*?(\d{1,2}\.\d{2})', page_text, re.IGNORECASE)
        if match:
            vix = float(match.group(1))
            logger.info(f"     ✅    VIX Found (Home Regex): {vix}")
            return round(vix, 2)
    except: pass

    logger.error("     ❌    VIX could not be found.")
    return 0.00

# --- TIMESTAMP EXTRACTOR ---
def extract_page_timestamp(page) -> str:
    try:
        src = page.content()
        match = re.search(r"As on\s+([0-9]{2}-[A-Za-z]{3}-[0-9]{4}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\s+IST)", src)
        if match:
            return match.group(1)
        return "Unknown"
    except:
        return "Error"

# --- SMART DROPDOWN FINDER ---
def find_expiry_dropdown(page):
    """
    ULTRA-SPECIFIC FINDER.
    It looks for a <select> element that specifically CONTAINS a date-like string.
    This prevents selecting the 'Symbol' dropdown by mistake.
    """
    # 1. Regex Filter Strategy (Most Reliable)
    try:
        # Define Regex for DD-Mon-YYYY
        date_pattern = re.compile(r"\d{1,2}-[A-Za-z]{3}-\d{4}")
        target = page.locator("select").filter(has_text=date_pattern).first
        
        if target.count() > 0 and target.is_visible():
            logger.info("   🎯   Found Expiry Dropdown via Content Match")
            return target
    except: pass

    # 2. ID Fallback
    try:
        loc = page.locator("#select_opt_expiry_date")
        if loc.count() > 0 and loc.is_visible():
             if loc.locator("option").count() > 1:
                return loc
    except: pass
    
    return None

def get_target_expiries(page, dropdown_element) -> Optional[Dict[str, List[str]]]:
    try:
        # Force a small wait and re-check content
        for _ in range(5):
            txt = dropdown_element.inner_text()
            if re.search(r"\d{4}", txt): # If it has a year, it's loaded
                break
            time.sleep(1)

        options = dropdown_element.locator("option").all()
        date_map = []
        
        for opt in options:
            val = opt.get_attribute("value")
            text = opt.inner_text()
            
            if "Select" in text or not val: continue
            try: 
                date_map.append((datetime.strptime(text, "%d-%b-%Y"), val))
            except: continue

        if not date_map: 
            return None
            
        date_map.sort(key=lambda x: x[0])
        
        month_groups = {}
        for d_obj, val in date_map:
            month_groups.setdefault((d_obj.year, d_obj.month), []).append((d_obj, val))

        monthlies = [month_groups[k][-1] for k in sorted(month_groups.keys())]
        quarterlies = [x for x in monthlies if x[0].month in [3, 6, 9, 12]]

        targets = {}
        def add_target(pair, label):
            d_obj, val = pair
            if val not in targets: targets[val] = {'date': d_obj, 'labels': []}
            if label not in targets[val]['labels']: targets[val]['labels'].append(label)

        if len(monthlies) >= 1: add_target(monthlies[0], 'm1')
        if len(monthlies) >= 2: add_target(monthlies[1], 'm2')
        if len(monthlies) >= 3: add_target(monthlies[2], 'm3')
        if len(quarterlies) >= 1: add_target(quarterlies[0], 'cq')
        if len(quarterlies) >= 2: add_target(quarterlies[1], 'nq')

        return targets
    except: return None

def switch_expiry(page, dropdown, value: str) -> bool:
    try:
        dropdown.evaluate(f"(el, val) => {{ el.value = val; el.dispatchEvent(new Event('change', {{ bubbles: true }})); }}", value)
        
        try:
            page.wait_for_selector(".loader", state="visible", timeout=2000)
            page.wait_for_selector(".loader", state="hidden", timeout=20000)
        except: pass
        
        page.wait_for_selector("#optionChainTable-indices", state="visible", timeout=20000)
        return True
    except: return False

def scrape_table_data(page) -> Optional[Dict]:
    try:
        try: 
            page.wait_for_selector("#optionChainTable-indices", timeout=20000)
        except: return None

        real_timestamp = extract_page_timestamp(page)
        spot = 0.0
        
        try:
            txt = page.locator("#equity_underlyingVal").inner_text()
            spot = float(re.search(r'([0-9,]+\.\d{2})', txt).group(1).replace(",", ""))
        except:
            content = page.content()
            match = re.search(r'NIFTY\s*(?:50)?\s*[:\s]\s*([0-9,]+\.\d{2})', content)
            if match: spot = float(match.group(1).replace(",", ""))
        
        if not validate_spot_price(spot): return None

        try:
            tbl_html = page.locator("#optionChainTable-indices").evaluate("el => el.outerHTML")
            dfs = pd.read_html(io.StringIO(tbl_html))
        except: return None
        
        df = next((d for d in dfs if d.shape[0] > 10), None)
        if df is None: return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

        strike_col = next((c for c in df.columns if "STRIKE" in str(c).upper()), None)
        iv_cols = [c for c in df.columns if "IV" in str(c).upper()]
        ltp_cols = [c for c in df.columns if "LTP" in str(c).upper()]

        if not strike_col or len(iv_cols) < 2: return None

        df = df[pd.to_numeric(df[strike_col], errors='coerce').notnull()].copy()
        df[strike_col] = df[strike_col].astype(float)
        df['diff'] = abs(df[strike_col] - spot)
        df_sorted = df.sort_values('diff').reset_index(drop=True)

        def safe(v):
            try: return float(v) if str(v).strip() != '-' else 0.0
            except: return 0.0

        atm = df_sorted.iloc[0]
        straddle = safe(atm[ltp_cols[0]]) + safe(atm[ltp_cols[1]])

        atm_call_iv, atm_put_iv = 0.0, 0.0
        for i in range(min(5, len(df_sorted))):
            c_iv, p_iv = safe(df_sorted.iloc[i][iv_cols[0]]), safe(df_sorted.iloc[i][iv_cols[1]])
            if c_iv > 0 and p_iv > 0:
                atm_call_iv, atm_put_iv = c_iv, p_iv
                break
        
        avg_iv = (atm_call_iv + atm_put_iv) / 2 if (atm_call_iv > 0 and atm_put_iv > 0) else 0.0
        skew_val, put_skew, call_skew = 0.0, 0.0, 0.0

        try:
            df_100 = df[df[strike_col] % 100 == 0].copy()
            def get_basket(center, col):
                targets = [center, center - 100, center + 100]
                vals = []
                for t in targets:
                    if not df_100.empty:
                        idx = (df_100[strike_col] - t).abs().idxmin()
                        v = safe(df_100.loc[idx][col])
                        if v > 0: vals.append(v)
                return sum(vals)/len(vals) if vals else 0.0

            put_skew = get_basket(spot * 0.98, iv_cols[1])
            call_skew = get_basket(spot * 1.02, iv_cols[0])
            if put_skew > 0 and call_skew > 0: skew_val = put_skew - call_skew
        except: pass

        return {
            "spot": spot, "straddle": straddle,
            "m1_call_iv": atm_call_iv, "m1_put_iv": atm_put_iv, "m1_iv": avg_iv,
            "skew_index": round(skew_val, 2),
            "skew_put_avg_9_10_11_iv": round(put_skew, 2),
            "skew_call_avg_9_10_11_iv": round(call_skew, 2),
            "nse_timestamp": real_timestamp
        }
    except: return None

# --- CORE LOGIC (SINGLE RUN) ---
def run_scrape_attempt(attempt_num: int) -> bool:
    logger.info(f"  🚀   Attempt {attempt_num}/3: Starting Scrape Process...")

    with sync_playwright() as p:
        try:
            browser = p.firefox.launch(headless=True, args=["--window-size=1920,1080"])
            
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
                viewport={'width': 1920, 'height': 1080},
                locale="en-US",
                timezone_id="Asia/Kolkata"
            )
            context.grant_permissions(['geolocation'])
            page = context.new_page()

            # 0. Check Permission
            if not check_scrape_permission(page):
                raise Exception("Access Denied or Market Closed")

            # 1. OPTION CHAIN NAVIGATION
            logger.info("   -> Navigating to Option Chain...")
            try:
                page.goto(OPTION_CHAIN_URL, wait_until="domcontentloaded", timeout=60000)
            except PlaywrightTimeoutError:
                logger.warning("   ⚠️   Page load timeout (DOM), checking if table exists...")

            stealth_setup(page)

            # Human stimulation
            try:
                page.wait_for_timeout(2000)
                page.mouse.move(100, 200)
                page.keyboard.press("PageDown")
                page.wait_for_timeout(500)
                page.keyboard.press("PageUp")
            except: pass

            logger.info("   -> Waiting for Option Chain Table...")
            try:
                page.wait_for_selector("#optionChainTable-indices", state="visible", timeout=45000)
            except PlaywrightTimeoutError:
                page.screenshot(path=f"debug_table_fail_{attempt_num}.png")
                raise Exception("Table #optionChainTable-indices not found (See screenshot)")

            logger.info("   -> Waiting for Dropdown Data...")
            try:
                 page.wait_for_selector("option:has-text('202')", timeout=15000)
            except: pass

            logger.info("   -> Searching for Expiry Dropdown...")
            dropdown = find_expiry_dropdown(page)
            
            if not dropdown: 
                page.screenshot(path=f"debug_dropdown_missing_{attempt_num}.png")
                raise Exception("Expiry Dropdown element could not be located.")

            # [LOGIC] Get Expiry Dates
            logger.info("   -> Extracting Expiry Targets...")
            targets = get_target_expiries(page, dropdown)
            
            if not targets:
                logger.warning("   ⚠️   Targets empty. Waiting and retrying...")
                time.sleep(3) 
                dropdown = find_expiry_dropdown(page)
                if dropdown:
                     targets = get_target_expiries(page, dropdown)
                
            if not targets:
                page.screenshot(path=f"debug_empty_options_{attempt_num}.png")
                raise Exception("No Expiry Targets Found (Options empty - See screenshot)")

            live_data = {}
            captured_exchange_time = None

            # [LOGIC] Iterate Targets
            for val, info in targets.items():
                labels, date_str = info['labels'], info['date'].strftime("%d-%b-%Y")
                logger.info(f"   -> Scraping: {date_str} {labels}")

                dropdown = find_expiry_dropdown(page)
                if dropdown:
                    switch_expiry(page, dropdown, val)
                    row_data = scrape_table_data(page)

                    if row_data:
                        if not captured_exchange_time and row_data.get('nse_timestamp') not in ["Unknown", "Error", None]:
                            captured_exchange_time = row_data['nse_timestamp']
                        
                        for lbl in labels:
                            if lbl == 'm1':
                                live_data.update(row_data)
                                live_data['m1_straddle'] = row_data['straddle']
                                live_data['m1_month'] = date_str
                            elif lbl == 'm2':
                                live_data.update({'m2_iv': row_data['m1_iv'], 'm2_straddle': row_data['straddle'], 'm2_month': date_str})
                            elif lbl == 'm3': live_data['m3_iv'] = row_data['m1_iv']
                            elif lbl == 'cq': live_data.update({'cq_iv': row_data['m1_iv'], 'cq_straddle': row_data['straddle'], 'cq_expiry': date_str})
                            elif lbl == 'nq': live_data.update({'nq_iv': row_data['m1_iv'], 'nq_straddle': row_data['straddle'], 'nq_expiry': date_str})

            # 2. VIX FETCH
            live_data['india_vix'] = get_india_vix_nse(page)

            if live_data.get('spot', 0) == 0:
                raise Exception("Spot Price is 0")

            # 3. SAVE TO DB
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()

            all_cols = [
                'spot_price', 'm1_straddle', 'm1_call_iv', 'm1_put_iv', 'm1_iv',
                'm2_iv', 'm3_iv', 'skew_put_avg_9_10_11_iv', 'skew_call_avg_9_10_11_iv',
                'skew_index', 'india_vix', 'm1_month', 'm2_month',
                'm2_straddle', 'cq_iv', 'nq_iv', 'cq_straddle', 'nq_straddle', 'cq_expiry', 'nq_expiry'
            ]

            live_data['spot_price'] = live_data.get('spot', 0)

            for k in all_cols:
                if k not in live_data:
                    live_data[k] = "" if ('month' in k or 'expiry' in k) else 0.00
                elif isinstance(live_data[k], float):
                    live_data[k] = round(live_data[k], 2)

            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if captured_exchange_time and "IST" in str(captured_exchange_time):
                try:
                    clean_ts = captured_exchange_time.replace(" IST", "").strip()
                    dt_obj = datetime.strptime(clean_ts, "%d-%b-%Y %H:%M:%S")
                    timestamp_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"  ⏰    Using Exchange Time: {timestamp_str}")
                except:
                    logger.warning("  ⚠️   Could not parse Exchange Time. Using Server Time.")
            
            today_date = timestamp_str.split(" ")[0]

            cursor.execute("DELETE FROM market_logs WHERE timestamp LIKE ?", (f"{today_date}%",))
            placeholders = ', '.join(['?'] * (len(all_cols) + 1))
            col_str = ', '.join(['timestamp'] + all_cols)
            values = [timestamp_str] + [live_data[c] for c in all_cols]

            cursor.execute(f"INSERT OR REPLACE INTO market_logs ({col_str}) VALUES ({placeholders})", values)
            conn.commit()
            conn.close()
            logger.info(f"  ✅    SUCCESS. Saved for {today_date}. (VIX: {live_data['india_vix']}, SPOT: {live_data['spot_price']})")
            
            browser.close()
            return True

        except Exception as e:
            logger.error(f"  ❌  Attempt {attempt_num} Failed: {e}")
            return False

# --- RETRY WRAPPER ---
def main_with_retries():
    init_db()

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        success = run_scrape_attempt(attempt)
        if success:
            logger.info("  🎉    Job Completed Successfully.")
            sys.exit(0)
        else:
            if attempt < max_retries:
                wait_time = 15
                logger.info(f"  ⏳    Waiting {wait_time}s before next attempt...")
                time.sleep(wait_time)
            else:
                logger.error("  💀    All attempts failed. Exiting.")
                sys.exit(1)

if __name__ == "__main__":
    main_with_retries()
