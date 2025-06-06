from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import json

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def format_title(title):
    title = title.replace('\n', '')
    while '  ' in title:
        title = title.replace('  ', ' ')
    return title


def get_case_links():
    with open('/Users/leedavis/coding/Python/SupremeCourtGuess/data/cases.json', 'r') as file:
        cases = json.loads(file.read())

    driver = webdriver.Chrome()

    for year in tqdm(range(1955, 2024)):
        url = f"https://www.oyez.org/cases/{year}"
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'h2.ng-scope')))
        html = driver.page_source

        soup = BeautifulSoup(html, "html.parser")

        case_headers = soup.find_all('h2', class_='ng-scope')
        for header in case_headers:
            case = header.find('a')
            cases.append({
                'name': format_title(case.text),
                'main_url': f'https://www.oyez.org/{case['href']}',
                'year': int(case['href'][6:10])
            })

    driver.quit()

    with open('/Users/leedavis/coding/Python/SupremeCourtGuess/data/cases.json', 'w') as file:
        file.write(json.dumps(cases))


get_case_links()
