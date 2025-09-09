from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
from urllib.parse import quote_plus
import itertools
import time


# Define Keywords and Queries

event_keywords = [
"massacre", "airstrike", "famine", "blockade", "rape", "looting",
"civilian casualties", "extrajudicial killings", "displacement","torture", "detention", "war crimes", "communication blackout",
"ethnic cleansing", "sexual violence", "gender-based violence","forced relocation", "collective punishment", "mass arrests, ”targeted killings", "indiscriminate bombing", "siege warfare",
"humanitarian crisis", "war crime allegations", "UN investigation","ICC referral", "peace talks failure", "aid obstruction",
"state violence", "government denial", "rebels blamed",
"accusations of genocide", "forced conscription", "child soldiers",
"aid blockade", "journalist killed", "access denied to journalists",
"information blackout", "media restriction", "foreign aid suspended" ] 

actor_keywords = [
    "TPLF", "Tigray People's Liberation Front", 
    "ENDF", "Ethiopian National Defense Force",
    "Eritrean forces", "Eritrean troops",
    "Amhara militia", "Fano militia", 
    "Ethiopian government", "Abiy Ahmed", 
    "Tigrayan rebels", "Ethiopian troops"
] 

context_keywords = [
    "Aksum", "Mai Kadra", "Mekelle", "Shire", "Adigrat", "Humera", "Alamata",
    "Western Tigray", "Southern Tigray", "Northern Ethiopia", "Central Tigray",
    "Kobo", "Wolkait", "Welkait", "Metekel", "Bahir Dar", "Gondar",
    "Amhara", "Afar", "Benishangul-Gumuz", "Oromia", "Eritrea", "Sudan border",
    "Tekeze River", "Djibouti corridor", "Red Sea access", "Eritrean border",
    "2020", "2021", "2022", "2023", "November 2020", "December 2021",
    "pre-ceasefire", "post-ceasefire", "peace agreement", "conflict escalation",
    "post-withdrawal", "pre-election", "post-election", "state of emergency",
    "Tigray conflict", "Ethiopia civil war", "armed confrontation", 
    "federal offensive", "military operation", "law and order campaign",
    "ethnic violence", "interethnic conflict", "rebel uprising",
    "insurgency", "federal intervention", "TPLF resurgence",
    "humanitarian corridor", "aid delivery routes", "food insecurity",
    "refugee camps", "IDPs", "cross-border displacement", "blocked aid",
    "UN convoy", "World Food Programme", "Ethiopian Red Cross",
    "international response", "UN condemnation", "AU mediation",
    "foreign intervention", "ceasefire violations", "diplomatic pressure"
]

event_str = "(" + " OR ".join(f'"{e}"' for e in event_keywords) + ")"
actor_str = "(" + " OR ".join(f'"{a}"' for a in actor_keywords) + ")"
context_str = "(" + " OR ".join(f'"{c}"' for c in context_keywords) + ")"

final_query = f"{event_str} AND {actor_str} AND {context_str}"


query_combos = list(itertools.product(event_keywords, actor_keywords, context_keywords))
queries = [f'({e}) AND ({a}) AND ({c})' for e, a, c in query_combos]

# Initialize Selenium WebDriver (Chrome example)
driver = webdriver.Chrome()


encoded_query = quote_plus(final_query)
base_url = f"https://advance.lexis.com/search/?pdmfid=1519360&crid=ef1b9429-e5a3-48e1-ad07-e6b248afabdb&pdsearchterms={encoded_query}&pdstartin=hlct%3A1%3A1&pdcaseshlctselectedbyuser=false&pdtypeofsearch=searchboxclick&pdsearchtype=SearchBox&pdtimeline=01%2F01%2F2019to31%2F10%2F2022%7Cdatebetween&pdoriginatingpage=bisnexishome&pdqttype=and&pdpsf=&pdquerytemplateid=&ecomp=wxbhkkk&earg=pdpsf&prid=0530628f-f65b-40d2-8710-3b7b9b7db38a"
driver.get(base_url)

input("🔐 Log in to the university portal manually. Once you're in and results are visible, press ENTER...")

wait = WebDriverWait(driver, 60)

with open('nexis_articles.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['title', 'author', 'source', 'date', 'url', 'query'])

    while True:
        # Wait for article blocks to appear
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[data-action="title"]')))
        articles = driver.find_elements(By.CSS_SELECTOR, 'a[data-action="title"]')

        for article in articles:
            try:
                title = article.text.strip()
                relative_url = article.get_attribute('href').strip()
                url = "https://advance.lexis.com" + relative_url if relative_url.startswith('/') else relative_url

                # Use relative paths to fetch metadata
                try:
                    parent_li = article.find_element(By.XPATH, './ancestor::li[1]')
                except:
                    parent_li = article.find_element(By.XPATH, './ancestor::*[contains(@class, "SearchResults")]')
                try:
                    author = parent_li.find_element(By.CSS_SELECTOR, 'a[data-type="nexis-props-author"]').text.strip()
                except:
                    author = ''

                try:
                    source = parent_li.find_element(By.CSS_SELECTOR, 'a[data-type="nexis-props-source"]').text.strip()
                except:
                    source = ''   
                try:
                    date = parent_li.find_element(By.CSS_SELECTOR, 'a[data-type="nexis-props-date"]').text.strip()
                except:
                    date = ''

                print(f'✅ {title} | {author} | {date}')
                writer.writerow([title, author, source, date, url])

            except Exception as e:
                print(f'⚠️ Error extracting article info: {e}')

        # Check for next page button
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, 'a.icon.la-TriangleRight.action[data-action="nextpage"]')
            if 'disabled' in next_btn.get_attribute('class'):
                print("🛑 No more pages.")
                break
            next_btn.click()
            time.sleep(3)
        except Exception as e:
            print(f'⚠️ Next button issue: {e}')
            break

driver.quit()