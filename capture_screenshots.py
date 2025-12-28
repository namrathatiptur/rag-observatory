"""Script to capture comprehensive dashboard screenshots using Selenium."""

import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

SCREENSHOTS_DIR = "screenshots"
DASHBOARD_URL = "http://localhost:8501"

def setup_driver():
    """Setup Chrome driver for screenshots."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        print("Please install ChromeDriver: brew install chromedriver")
        return None

def capture_screenshot(driver, filename, wait_seconds=3):
    """Capture a screenshot and save it."""
    try:
        time.sleep(wait_seconds)
        screenshot_path = os.path.join(SCREENSHOTS_DIR, filename)
        driver.save_screenshot(screenshot_path)
        print(f"✓ Captured: {filename}")
        return True
    except Exception as e:
        print(f"✗ Error capturing {filename}: {e}")
        return False

def wait_for_element(driver, by, value, timeout=10):
    """Wait for an element to be present."""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except:
        return None

def click_element(driver, element):
    """Click an element with retry."""
    try:
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", element)
        time.sleep(1)
        return True
    except:
        return False

def main():
    """Main function to capture dashboard screenshots."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    
    driver = setup_driver()
    if not driver:
        print("\nManual screenshot instructions:")
        print("1. Open http://localhost:8501 in your browser")
        print("2. Use Cmd+Shift+4 to take screenshots")
        print("3. Save to screenshots/ directory")
        return
    
    try:
        print(f"\nOpening dashboard: {DASHBOARD_URL}")
        driver.get(DASHBOARD_URL)
        
        # Wait for dashboard to load
        print("Waiting for dashboard to load...")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(8)  # Give Streamlit time to fully render
        
        # Screenshot 1: Main dashboard view
        print("\n1. Capturing main dashboard view...")
        capture_screenshot(driver, "01_main_dashboard.png", 2)
        
        # Screenshot 2: Configuration sidebar
        print("\n2. Capturing configuration sidebar...")
        capture_screenshot(driver, "02_configuration_sidebar.png", 2)
        
        # Try to find and interact with mode selection
        try:
            # Look for radio buttons or mode selection
            labels = driver.find_elements(By.TAG_NAME, "label")
            if labels:
                print("\n3. Capturing mode selection...")
                capture_screenshot(driver, "03_mode_selection.png", 2)
        except Exception as e:
            print(f"Could not capture mode selection: {e}")
        
        # Scroll down to see more content
        print("\n4. Scrolling and capturing full view...")
        driver.execute_script("window.scrollTo(0, 500);")
        time.sleep(2)
        capture_screenshot(driver, "04_scrolled_view.png", 2)
        
        # Try to find tabs
        try:
            tabs = driver.find_elements(By.CSS_SELECTOR, "[role='tab']")
            if tabs:
                print(f"\n5. Found {len(tabs)} tabs, capturing each...")
                for i, tab in enumerate(tabs[:4]):  # Capture first 4 tabs
                    if click_element(driver, tab):
                        time.sleep(3)
                        capture_screenshot(driver, f"05_tab_{i+1}.png", 2)
        except Exception as e:
            print(f"Could not capture tabs: {e}")
        
        # Full page screenshot
        print("\n6. Capturing full page...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        capture_screenshot(driver, "06_full_page.png", 2)
        
        # Scroll back to top
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        capture_screenshot(driver, "07_top_view.png", 2)
        
        print(f"\n✓ All screenshots saved to: {os.path.abspath(SCREENSHOTS_DIR)}/")
        print(f"  Total files: {len(os.listdir(SCREENSHOTS_DIR))}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
