
import json
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from tqdm import tqdm


def get_transcript(url):
    file_path = os.path.join(DATA_DIR, 'raw_cases', f"{url.replace('/', '-')}.json")

    if os.path.exists(file_path):
        # print(f"File {file_path} already exists. Skipping.")
        return

    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'section.transcript-turn.ng-scope')))

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    sections = soup.find_all("section", class_='transcript-turn ng-scope')

    statements = []

    for section in sections:
        text = ' '.join([p.text for p in section.find_all('p')])
        statements.append({
            'speaker': (section.find('h4').text),
            'text': (text)
        })

    with open(file_path, 'w') as file:
        file.write(json.dumps(statements))


with open(os.path.join(DATA_DIR, 'basic.json'), 'r') as file:
    main = json.load(file)

# Pre-filter cases that have transcript URLs
cases_with_transcripts = []
cases_without_transcripts = []

for case in main:
    transcript_url = case.get('transcript_url')
    if transcript_url:
        cases_with_transcripts.append(case)
    else:
        cases_without_transcripts.append(case)

print(f"Total cases: {len(main)}")
print(f"Cases with transcript URLs: {len(cases_with_transcripts)}")
print(f"Cases without transcript URLs: {len(cases_without_transcripts)}")

if not cases_with_transcripts:
    print("No cases with transcript URLs found. Exiting.")
    exit()

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--remote-debugging-port=9222')

driver = webdriver.Chrome(options=chrome_options)

for case in tqdm(cases_with_transcripts, desc="Processing transcripts"):
    try:
        transcript_url = case.get('transcript_url')
        get_transcript(transcript_url)
    except TimeoutException:
        print(
            f"Timeout error for URL {transcript_url}: Page took too long to load")
    except WebDriverException as e:
        print(f"WebDriver error for URL {transcript_url}: {e}")
    except Exception as e:
        print(f"Error processing case {case}: {e}")
        break

driver.quit()
