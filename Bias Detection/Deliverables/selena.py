from seleniumwire import webdriver  # selenium-wire import
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/amalkurian/Desktop/Dissertation/Bias Detection/env/config.env", override=True)
email = os.getenv("Email")
password = os.getenv("PASSWORD")

from selenium.common.exceptions import (
    InvalidSessionIdException,
    ElementClickInterceptedException,
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)
import csv
import pandas as pd
import time
import logging

# Configure logger
logging.basicConfig(
    filename='batch_report.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Base URL for Nexis Uni search (your current search URL)
base_url = ("https://advance.lexis.com/search/?pdmfid=1519360&crid=5349c729-0525-46a6-986d-3cf698c87c01"
            "&pdsearchtype=SearchBox&pdtypeofsearch=searchboxclick&pdstartin=urn%3Ahlct%3A16"
            "&pdsearchterms=Ethiopian+Tigary+War&pdtimeline=31%2F07%2F2019+to+31%2F10%2F2022%7Cbetween%7CDD%2FMM%2FYYYY"
            "&pdpost=urn%3Ahlct%3A16%2CTmV3c3BhcGVycw%3D%3D~TmV3c3dpcmVzICYgUHJlc3MgUmVsZWFzZXM%3D~SW5kdXN0cnkgVHJhZGUgUHJlc3M%3D~TmV3cyBUcmFuc2NyaXB0cw%3D%3D~V2ViTGlua3M%3D~QmxvZ3M%3D%5Epublicationtype~publicationtype~publicationtype~publicationtype~publicationtype~publicationtype%5ENewspapers%3ANewswires%2520%2526%2520Press%2520Releases%3AIndustry%2520Trade%2520Press%3ATranscripts%3AWebnews%3ABlogs%5E%5Etrue"
            "&ecomp=wxbhkkk&prid=3b18de21-6b79-469a-b39b-4859412b0700")

# Keywords as per your lists
event_keywords = [
    "Ethiopia War","Togoga market massacre","Hawzen rape camp",
    "Shire shelling","mass grave burning",
    "Ethnic cleansing", "TPLF resurgence","Tigray Conflict", 
    "Diplomatic pressure", "War crimes allegations", "Ethiopian Tigray War",
    "arbitrary detention","human trafficking", "war crimes","ethnic cleansing", "disappearances",
    "clandestine operations",  "Tigray War", 
    "water shortages","Togoga airstrike","Federal offensive",
    "Eritrean forces", "Pre Election", "Post Election",
    "Post-ceasefire", "Post-withdrawal", "Refugee camps", "IDPs"
]

actor_keywords = [
    "TPLF", "Tigray People's Liberation Front","Zalambessa killings",
    "Hawzen executions","ENDF", "Ethiopian National Defense Force",
    "Amhara militia", "Fano militia",  "Dengelat massacre",
    "Maryam Dengelat church", "Galikoma massacre",
    "Ethiopian government", "Abiy Ahmed", "Ethiopian troops"
]

CHROME_OPTS = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']


# Initialize driver and wait
def init_driver():
    opts = webdriver.ChromeOptions()
    for arg in CHROME_OPTS: opts.add_argument(arg)
    return webdriver.Chrome(service=Service(), options=opts,
                            seleniumwire_options={'connection_timeout': None})

# Helper functions

def wait_for_spinner_to_clear(wait, timeout=40):
    """Wait until loading spinner (aria-busy) disappears"""
    try:
        wait.until_not(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[aria-busy="true"]')))
    except TimeoutException:
        print("⚠️ Spinner did not disappear within timeout.")

def safe_click(driver, element, max_attempts=5, wait_time=1):
    """Try clicking element several times, scrolling into view first"""
    for attempt in range(max_attempts):
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(2)
            element.click()
            return
        except ElementClickInterceptedException as e:
            print(f"⚠️ Click attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise Exception("Element not clickable after multiple attempts.")

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# Open the page and wait for user to log in manually
driver = init_driver()
driver.get(base_url)
wait = WebDriverWait(driver, 20)
email_input = wait.until(EC.presence_of_element_located((By.ID, "userid")))
# Clear the field
email_input.clear()
email_input.send_keys(email)

next_button = wait.until(EC.element_to_be_clickable((By.ID, "signInSbmtBtn")))
next_button.click()

time.sleep(5)

pass_input = wait.until(EC.element_to_be_clickable((By.ID, "password")))
pass_input.clear()
pass_input.send_keys(password)

# Step 4: Click Sign In
sign_in_btn = wait.until(EC.element_to_be_clickable((By.ID, "next")))
sign_in_btn.click()
print("\nLogged IN!")
#input("🔐 Log in manually and wait for results to load, then press ENTER here...")


df = pd.read_csv(f"/Users/amalkurian/Desktop/Dissertation/Bias Detection/nexis_articles1.csv")
batch_size = 5
seen_titles = set(df['title'])


with open('nexis_articles2.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['title', 'author', 'source', 'date', 'url', 'query'])

    batch_num = 0

    for actor_batch in chunk_list(actor_keywords, batch_size):
        for event_batch in chunk_list(event_keywords, batch_size):
            event_str = "(" + " OR ".join(f'"{e}"' for e in event_batch) + ")"
            actor_str = "(" + " OR ".join(f'"{a}"' for a in actor_batch) + ")"
            final_query = f"{event_str} AND {actor_str}"

            try:
                # Enter query in search box
                search_box = wait.until(EC.presence_of_element_located((By.ID, "searchTerms")))
                wait_for_spinner_to_clear(wait)
                safe_click(driver, search_box)
                search_box.send_keys(Keys.COMMAND, 'a')
                search_box.send_keys(Keys.BACKSPACE)
                time.sleep(3)
                search_box.send_keys(final_query)
                time.sleep(5)
                search_box.send_keys(Keys.ENTER)

                
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'h2.doc-title.translate')))
                time.sleep(5)

                articles_collected = 0

                while True:
                    try:
                        titles = driver.find_elements(By.XPATH, "//h2[contains(@class, 'doc-title translate')]//a[@data-action='title']")
                        driver.refresh()
                        for i, a_tag in enumerate(titles):
                            try:
                                time.sleep(5)
                                # Refetch titles list to avoid stale elements
                                titles = driver.find_elements(By.XPATH, "//h2[contains(@class, 'doc-title translate')]//a[@data-action='title']")
                                a_tag = titles[i]
                                title_text = a_tag.text.strip()

                                # Extract metadata
                                author = ''
                                source = ''
                                date = ''
                                article_url = ''


                                if title_text not in seen_titles:
                                    seen_titles.add(title_text)
                                else:
                                    continue



                                try:
                                    parent_li = a_tag.find_element(By.XPATH, './ancestor::li[1]')
                                    try:
                                        author = parent_li.find_element(By.CSS_SELECTOR, 'a[data-type="nexis-props-author"]').text.strip()
                                    except:
                                        pass
                                    try:
                                        source = parent_li.find_element(By.CSS_SELECTOR, 'a[data-type="nexis-props-source"]').text.strip()
                                    except:
                                        pass
                                    try:
                                        date = parent_li.find_element(By.CSS_SELECTOR, 'a[data-type="nexis-props-date"]').text.strip()
                                    except:
                                        pass
                                except:
                                    pass
                                

                                teaser_link = parent_li.find_element(By.CSS_SELECTOR, 'a[data-action="hitsteaser"]')
                                href_raw = teaser_link.get_attribute("href")
                                href_resolved = driver.execute_script("return arguments[0].href;", teaser_link)

                                if href_raw and "/teaserdocument/" in href_raw:
                                    article_url = href_raw
                                elif href_resolved and "/teaserdocument/" in href_resolved:
                                    article_url = href_resolved
                                else:
                                    # Fallback to click + network sniffing
                                    # Clear previous requests before click
                                    driver.requests.clear()

                                    # Scroll into view & click using JS to avoid click intercept issues
                                    driver.execute_script("arguments[0].scrollIntoView(true);", a_tag)
                                    driver.execute_script("arguments[0].click();", teaser_link)
                                    
                                    WebDriverWait(driver, 10).until(EC.url_contains("/document/"))
                                    time.sleep(2)
                                    # Sniff network requests for real article URL
                                    article_url = driver.current_url

                                    # const a = document.querySelector('a[data-action="hitsteaser"][data-idx="0"]');
                                    # console.log(a.getAttribute('href'));
                                    # console.log(a.href);

                                    candidate_urls = [
                                        req.url for req in driver.requests
                                        if req.response
                                        and "/document/" in req.url
                                    ]
                                    # Fallback to current_url if not found
                                    if candidate_urls:
                                        article_url = candidate_urls[-1]  # last one is most likely full article
                                    else:
                                        article_url = driver.current_url

                                    driver.back()
                                    
                                print(f"{i}. ✅ CLICKED | Title: {title_text} | URL sniffed: {article_url}")
                                writer.writerow([title_text, author, source, date, article_url, final_query])
                                articles_collected += 1

                                
                                wait_for_spinner_to_clear(wait)
                                time.sleep(2)
                                wait.until(EC.presence_of_element_located((By.ID, 'results-list-delivery-toolbar')))

                            except Exception as e:
                                print(f'⚠️ Article extraction error at index {i}: {e}')

                        if articles_collected >= 700:
                            break

                        # Pagination - next page
                        try:
                            next_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.icon.la-TriangleRight.action[data-action="nextpage"]')))
                            if 'disabled' in next_btn.get_attribute('class'):
                                print("🛑 Reached last page.")
                                break
                            else:
                                wait_for_spinner_to_clear(wait, timeout=20)
                                driver.execute_script("arguments[0].scrollIntoView(true);", next_btn)
                                safe_click(driver, next_btn)
                                print("➡️ Moved to next page.")
                                time.sleep(4)
                        except Exception as e:
                            print(f'⚠️ Next page button issue or no more pages: {e}')
                            driver.refresh()
                            break

                    except Exception as e:
                        driver.refresh()
                        print(f'⚠️ Error on page or loading articles: {e}')
                        break

                log_msg = (f"Batch {batch_num} | Events: {event_batch} | Actors: {actor_batch} | "
                           f"Query: \"{final_query}\" | Articles Collected: {articles_collected}")
                batch_num += 1

                if articles_collected == 0:
                    logging.warning("⚠️ " + log_msg)
                elif articles_collected < 10:
                    logging.info("🔍 " + log_msg)
                else:
                    logging.info("✅ " + log_msg)

                print(f'🔍 {final_query} yielded {articles_collected} articles')
                driver.refresh()
            except Exception as e:
                print(f"⚠️ Error in batch processing: {e}")

driver.quit()
