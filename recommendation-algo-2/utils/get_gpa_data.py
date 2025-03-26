import csv
import logging
import time
import random

from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ----------------------------------------------------------------------------
# 1. CONFIGURE LOGGING
# ----------------------------------------------------------------------------
# This sets up a basic logger that prints to the console and also writes to a file.
logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more verbose output
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scrape_gpa.log", mode="w"),  # logs go to file
        logging.StreamHandler()                           # logs also to console
    ]
)

# ----------------------------------------------------------------------------
# 2. SETUP SELENIUM
# ----------------------------------------------------------------------------
try:
    driver = webdriver.Chrome()  # Or webdriver.Firefox(), etc.
except WebDriverException as e:
    logging.error(f"Could not start the WebDriver: {e}")
    raise SystemExit

# The main page that lists the GPA links (update to your real URL)
MAIN_URL = "https://www.appily.com/colleges/gpa/"

try:
    driver.get(MAIN_URL)
    # Give a little time for the page to load
    time.sleep(3)
except Exception as e:
    logging.error(f"Failed to load main page {MAIN_URL}: {e}")
    driver.quit()
    raise SystemExit

# ----------------------------------------------------------------------------
# 3. COLLECT ALL GPA LINKS
# ----------------------------------------------------------------------------
try:
    # Wait up to 10 seconds for the container of GPA links to be present
    gpa_container = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.college-list--by-gpa ul"))
    )
    gpa_links = gpa_container.find_elements(By.CSS_SELECTOR, "li a")
    gpa_urls = [link.get_attribute("href") for link in gpa_links]
    logging.info(f"Found {len(gpa_urls)} GPA URLs.")
except TimeoutException:
    logging.error("Timed out waiting for GPA links to appear on the main page.")
    driver.quit()
    raise SystemExit
except Exception as e:
    logging.error(f"Error while gathering GPA links: {e}")
    driver.quit()
    raise SystemExit


# ----------------------------------------------------------------------------
# 4. SCRAPE SCHOOLS ON ONE PAGE
# ----------------------------------------------------------------------------
def scrape_gpa_page():
    """
    Scrapes all school cards on the current page and returns
    a list of dictionaries with the desired data.
    """
    school_data = []
    try:
        articles = driver.find_elements(By.CSS_SELECTOR, "article.college-list--card.gpa-result")
    except Exception as e:
        logging.warning(f"Could not find any articles with 'college-list--card.gpa-result': {e}")
        return school_data

    for art in articles:
        row = {}

        # School name
        try:
            name_el = art.find_element(By.CSS_SELECTOR, "div.college-list--card-title-conatiner a")
            row["name"] = name_el.text.strip()
        except NoSuchElementException:
            row["name"] = ""
        except Exception as e:
            logging.debug(f"Error getting 'name': {e}")
            row["name"] = ""

        # Location
        try:
            location_el = art.find_element(By.CSS_SELECTOR, "div.college-list--card-location")
            row["location"] = location_el.text.strip()
        except NoSuchElementException:
            row["location"] = ""
        except Exception as e:
            logging.debug(f"Error getting 'location': {e}")
            row["location"] = ""

        # Average GPA
        try:
            avg_gpa_el = art.find_element(By.CSS_SELECTOR, "div.field.average-gpa")
            row["average_gpa"] = avg_gpa_el.text.strip()
        except NoSuchElementException:
            row["average_gpa"] = ""
        except Exception as e:
            logging.debug(f"Error getting 'average_gpa': {e}")
            row["average_gpa"] = ""

        # Acceptance Rate
        try:
            acceptance_rate_el = art.find_element(By.CSS_SELECTOR, "div.field.acceptance-rate")
            row["acceptance_rate"] = acceptance_rate_el.text.strip()
        except NoSuchElementException:
            row["acceptance_rate"] = ""
        except Exception as e:
            logging.debug(f"Error getting 'acceptance_rate': {e}")
            row["acceptance_rate"] = ""

        # Helper function to find data by label in the card
        def get_data_by_label(label_text):
            try:
                parent = art.find_element(
                    By.XPATH,
                    f'.//div[contains(@class,"college-list--card-outer") and .//div[contains(text(),"{label_text}")]]'
                )
                return parent.find_element(By.CSS_SELECTOR, 'div.college-list--card-data-val').text.strip()
            except NoSuchElementException:
                return ""
            except Exception as e:
                logging.debug(f"Error getting label '{label_text}': {e}")
                return ""

        row["average_act_composite"] = get_data_by_label("average act composite")
        row["average_sat_composite"] = get_data_by_label("average sat composite")
        row["type_of_institution"] = get_data_by_label("type of institution")
        row["level_of_institution"] = get_data_by_label("level of institution")
        row["average_net_price"] = get_data_by_label("average net price")
        row["number_of_students"] = get_data_by_label("number of students")

        school_data.append(row)

    return school_data


# ----------------------------------------------------------------------------
# 5. SCRAPE EACH GPA PAGE WITH PAGINATION & WRITE CSV
# ----------------------------------------------------------------------------
csv_filename = "colleges_data.csv"

# Write the CSV header row once
with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "gpa_link",
        "name",
        "location",
        "average_gpa",
        "acceptance_rate",
        "average_act_composite",
        "average_sat_composite",
        "type_of_institution",
        "level_of_institution",
        "average_net_price",
        "number_of_students"
    ])

# Open again in append mode so we can write page by page
with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # Loop over each GPA link
    for url in gpa_urls:
        logging.info(f"Scraping GPA page: {url}")

        # Go to that GPA page
        try:
            driver.get(url)
            time.sleep(random.uniform(2, 4))  # random delay
        except Exception as e:
            logging.error(f"Failed to load {url}: {e}")
            # Optionally prompt user if they want to continue or skip
            user_input = input("Failed to load page. Press Enter to retry or type 'skip' to skip: ")
            if user_input.lower() == "skip":
                continue
            else:
                # attempt second try
                try:
                    driver.get(url)
                    time.sleep(random.uniform(2, 4))
                except:
                    logging.error(f"Second attempt failed for {url}, skipping.")
                    continue

        # Keep paginating until we can’t
        while True:
            # Scrape schools on the current page
            rows = scrape_gpa_page()

            # Write them to CSV
            for row in rows:
                writer.writerow([
                    url,
                    row["name"],
                    row["location"],
                    row["average_gpa"],
                    row["acceptance_rate"],
                    row["average_act_composite"],
                    row["average_sat_composite"],
                    row["type_of_institution"],
                    row["level_of_institution"],
                    row["average_net_price"],
                    row["number_of_students"]
                ])

            # Try to click the "Next" button
            try:
                # Wait for the next button to be clickable (if it exists)
                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "li.pager__item.pager__item--next a"))
                )
                next_button.click()
                logging.info("Clicked Next button.")
                time.sleep(random.uniform(2, 4))  # random delay
            except TimeoutException:
                logging.info("No Next button found (likely last page).")
                break
            except Exception as e:
                logging.warning(f"Error clicking Next button: {e}")
                # Give user a chance to see what's happening
                user_input = input("Press Enter to retry clicking Next or type 'skip' to move on: ")
                if user_input.lower() == "skip":
                    break
                else:
                    # Try once more
                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, "li.pager__item.pager__item--next a")
                        next_button.click()
                        logging.info("Retried clicking Next button successfully.")
                        time.sleep(random.uniform(2, 4))
                    except Exception as e2:
                        logging.error(f"Second attempt to click Next failed: {e2}")
                        break

logging.info("Scraping complete. Data saved to colleges_data.csv")

# ----------------------------------------------------------------------------
# 6. CLEANUP
# ----------------------------------------------------------------------------
driver.quit()
