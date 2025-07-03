
NEXIS DATABASE

Usage

Clone the repo

Add your Nexis Uni login credentials to a .env file (Email and PASSWORD)

Install dependencies from requirements.txt

Run python selena.py to start the scraper
(python selena.py)

Output CSVs will be saved with article metadata for downstream bias detection tasks


Features

Automated Login
Credentials are securely loaded from a .env file using dotenv (Email and PASSWORD keys).

Query Batching
Combines event_keywords and actor_keywords in small batches to run logical AND searches like:
("War crimes" OR "IDPs") AND ("ENDF" OR "Abiy Ahmed").

Fully Automated Interaction
Handles JavaScript-heavy elements, waits for spinners, navigates 
pagination, and retries when elements fail.

Rich Metadata Extraction
Extracts:
title
author
source
publication date
article URL
query used

Incremental CSV Storage
Results are written to nexis_articles2.csv as they’re scraped — no risk of losing data mid-batch.

Batch Logging
Outputs batch-level success logs and error logs to batch_report.log
Includes query info, count of articles collected, and warnings for empty results





