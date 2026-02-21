from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
import json
import os
import requests
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

with open(os.path.join(DATA_DIR, 'cases.json'), 'r') as file:
    cases = json.load(file)


def get_basic(case):
    """
    Extract basic case information from Supreme Court case page.

    Args:
        case (dict): Case dictionary containing 'main_url', 'name', and 'year'

    Returns:
        dict: Extracted case information including transcript URL, petitioners, 
              facts, question, votes, majority, and parties
        None: If case was vacated and remanded
    """
    url = case['main_url']
    driver.get(url)

    # Wait for the page to load completely
    WebDriverWait(driver, 3).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'h3.vote-description'))
    )

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # Check conclusion first - return None if vacated and remanded
    conclusion = _extract_conclusion(soup)
    if conclusion is not None and 'Vacated and remanded' in conclusion:
        return None

    # Extract transcript URL
    transcript_url = _extract_transcript_url(soup)

    # Extract petitioners information
    petitioners = _extract_petitioners(soup)

    # Extract case facts
    facts = _extract_facts(soup)

    # Extract question
    question = _extract_question(soup)

    # Extract votes and majority
    votes = _extract_votes(soup)
    majority = _extract_majority(soup)

    # Extract parties
    parties = _extract_parties(soup)

    return {
        'name': case['name'],
        'main_url': url,
        'year': case['year'],
        'transcript_url': transcript_url,
        'petitioners': petitioners,
        'facts': facts,
        'question': question,
        'votes': votes,
        'majority': majority,
        'parties': parties
    }


def _extract_conclusion(soup):
    """Extract conclusion from the page."""
    conclusion_div = soup.find('div', attrs={'ng-if': 'case.conclusion'})
    if conclusion_div:
        # Look for conclusion text in various possible elements
        conclusion_text = conclusion_div.get_text(strip=True)
        if conclusion_text:
            return conclusion_text
    return None


def _extract_transcript_url(soup):
    """Extract transcript URL from the page."""
    li = soup.find(
        'li', attrs={'ng-repeat': 'audio in case.oral_argument_audio'})
    if li:
        a = li.find('a')
        if a and 'iframe-url' in a.attrs:
            return a.attrs['iframe-url']
    return ''


def _extract_petitioners(soup):
    """Extract petitioners information from the page."""
    divs = soup.find_all('div', class_='advocate ng-scope')
    petitioners = []
    for div in divs:
        a_tag = div.find('a')
        span_tag = div.find('span')
        if a_tag and span_tag:
            petitioners.append({
                'name': a_tag.text.strip(),
                'for': span_tag.text.strip()
            })
    return petitioners


def _extract_facts(soup):
    """Extract case facts from the page."""
    facts_div = soup.find(
        'div', attrs={'ng-bind-html': 'case.facts_of_the_case'})
    if facts_div:
        paragraphs = facts_div.find_all('p')
        if paragraphs:
            return '\n'.join([p.text.strip() for p in paragraphs])
    return ''


def _extract_question(soup):
    """Extract question from the page."""
    question_div = soup.find('div', attrs={'ng-bind-html': 'case.question'})
    if question_div:
        p_tag = question_div.find('p')
        if p_tag:
            return p_tag.text.strip()
    return ''


def _extract_votes(soup):
    """Extract votes information from the page."""
    vote_figures = soup.find_all(
        'figure', attrs={'ng-class': '[vote.vote, vote.orderClass]'})
    votes = []
    for vote in vote_figures:
        name_span = vote.find('span', class_='long ng-binding')
        if name_span and 'class' in vote.attrs:
            votes.append({
                'name': name_span.text.strip(),
                'vote': 'majority' if 'majority' in vote.attrs['class'] else 'minority'
            })
    return votes


def _extract_majority(soup):
    """Extract majority information from the page."""
    majority_span = soup.find('span', class_='winner ng-binding ng-scope')
    if majority_span:
        return majority_span.text.strip()
    return ''


def _extract_parties(soup):
    """Extract parties information from the page."""
    aside = soup.find('aside')
    if aside:
        row = aside.find('div', class_='row')
        if row:
            party_divs = row.find_all('div')
            parties = {}
            for party_div in party_divs:
                text = party_div.text.strip()
                if text:
                    words = text.split()
                    if len(words) >= 2:
                        parties[words[0]] = ' '.join(words[1:])
            return parties
    return {}


# Configuration: Set to True to re-run previously failed cases
RERUN_BAD_CASES = False

with open(os.path.join(DATA_DIR, 'basic.json'), 'r') as file:
    main = json.load(file)

with open(os.path.join(DATA_DIR, 'bad_cases.json'), 'r') as file:
    bad_cases = json.load(file)

# Filter out cases that are already processed
existing_case_names = {case['name'] for case in main}
bad_case_names = {case['name'] for case in bad_cases}

if RERUN_BAD_CASES:
    # Include bad cases in processing if rerun is enabled
    cases_to_process = [
        case for case in cases if case['name'] not in existing_case_names]
else:
    # Exclude bad cases from processing
    cases_to_process = [case for case in cases if case['name']
                        not in existing_case_names and case['name'] not in bad_case_names]

print(
    f"Total cases: {len(cases)}, Already processed: {len(existing_case_names)}, Bad cases: {len(bad_case_names)}, To process: {len(cases_to_process)}, Rerun bad cases: {RERUN_BAD_CASES}")

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--remote-debugging-port=9222')

driver = webdriver.Chrome(options=chrome_options)

for case in tqdm(cases_to_process, desc="Processing cases"):
    try:
        case_data = get_basic(case)
        if case_data is not None:  # Only append if not vacated and remanded
            main.append(case_data)

            # If this case was previously in bad_cases and now processed successfully, remove it
            if RERUN_BAD_CASES and case['name'] in bad_case_names:
                bad_cases = [
                    bc for bc in bad_cases if bc['name'] != case['name']]
                with open(os.path.join(DATA_DIR, 'bad_cases.json'), 'w') as file:
                    json.dump(bad_cases, file)
                print(
                    f"Successfully processed previously failed case: {case['name']}")

            with open(os.path.join(DATA_DIR, 'basic.json'), 'w') as file:
                json.dump(main, file)
    except TimeoutException:
        # WebDriverWait timeout - add to bad cases and continue
        print(f"Timeout waiting for page elements for case: {case['name']}")
        # Only add to bad_cases if it's not already there
        if case['name'] not in bad_case_names:
            bad_cases.append(case)
        with open(os.path.join(DATA_DIR, 'bad_cases.json'), 'w') as file:
            json.dump(bad_cases, file)
        continue
    except Exception as e:
        print(
            f"An error occurred while processing case {case['main_url']}: {e}")
        break

driver.quit()
