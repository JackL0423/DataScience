# script for reading in normal lines and spitting out ones that match expression
# egrep.py
from multiprocessing import process
import sys, re

# sys.argv is the list of command-line arguments
# sys.argv[0] is the name of the program itself
# sys.argv[1] will be the regex specified at the command line
regex = sys.argv[1]

# for every line passed into the script 
for line in sys.stdin:
    # if it matches the regex, write it to stdout
    if re.search(regex, line):
        sys.stdout.write(line)

# script for spitting out number of lines in file
# line_count.py
import sys

count = 0
for line in sys.stdin:
    count += 1

# print goes to sys.stdout
print(count)

# script that counts words in its input and writes out the most common ones
# most_common_words.py
import sys
from collections import Counter

# pass in number of words as first argument
try:
    num_words = int(sys.argv[1])
except:
    print("usage: most_common_words,py num_words")
    sys.exit(1)     # nonzero exit code

counter = Counter(word.lower()
                  for line in sys.stdin
                  for word in line.strip().split())

for word, count in counter.most_common(num_words):
    sys.stdout.write(str(count))
    sys.stdout.write("\t")
    sys.stdout.write(word)
    sys.stdout.write("\n")

# 'r' means read-only, it's assumed if you leave it out
#file_for_reading = open('reading_file.txt', 'r')
#file_for_reading2 = open('reading_file.txt')

# 'w' us write -- will destroy the file if it already exists
file_for_writing = open('writing_file.txt', 'w')

# 'a' is append -- for adding to the end of the file
#file_for_appending = open('appending_file.txt', 'a')

# need to close files after use
file_for_writing.close()

# better way to interact with files...
#with open(filename) as f:
#    data = function_to_get_data_from(f)

# file is closed after done.

# if you need an entire text file you can iterate over it
starts_with_hash = 0

with open('input.txt') as f:
    if re.match("^#", line):
        starts_with_hash += 1


def get_domain(email_address: str) -> str:
    """Split on '@' and return the last piece"""
    return email_address.lower().split("@")[-1]

# a couple of tests
#assert get_domain('joelgrus@gmail.com') == 'gmail.com'
#assert get_domain('hoel@m.datasciencester.com') == 'm.datascciencester.com'

# with open('email_addresses.txt', 'r') as f:
    # domain_counts = Counter(get_domain(line.strip())
                            # for line in f
                            # if "@" in line)

# Delimiting Files
import csv

with open('tab_delimited_stock_prices.txt') as f:
    tab_reader = csv.reader(f, delimeter='\t')
    for row in tab_reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(date, symbol, closing_price)

with open('colon_delimited_stock_prices.txt') as f:
    colon_reader = csv.DictReader(f, delimiter=':')
    for dict_row in colon_reader:
        date = dict_row["date"]
        symbol = dict_row["symbol"]
        closing_price = float(dict_row["closing_price"])
        process(date, symbol, closing_price)

# HTML and Parsing Thereof
from bs4 import BeautifulSoup
import requests

url = ("https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html")
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

first_paragraph = soup.find('p')    # or just soup.p
first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()

# extract a tag's attributes
first_paragraph_id = soup.p['id']           # raises KeyError if no 'id'
first_paragraph_id2 = soup.p.get('id')      # returns None is no 'id'

# getting multiple tags at once
all_paragraphs = soup.find_all('p')     # or just soup('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]

# getting classes with specific classes
important_paragraphs = soup('p', {'class' : 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p') if 'important' in p.get('class', [])]

url = "https://www.house.gov/representatives"
text = requests.get(url)
soup = BeautifulSoup(text, "html5lib")

all_urls = [a['href'] for a in soup('a') if a.has_attr('href')]

#good_urls = list(set(good_urls))

# using unauthenticated API
import requests, json

github_user = "joelgrus"
endpoint = f"https://api.github.com/users/{github_user}/repos"

repos = json.loads(requests.get(endpoint).text)

from collections import Counter
from dateutil.parser import parse

dates = [parse(repo["created_at"]) for repo in repos]
month_counts = Counter(date.month for date in dates)
weekday_counts = Counter(date.weekday() for date in dates)

last_5_repositories = sorted(repos, key=lambda r: r["pushed_at"], reverse=True)[:5]

last_5_languages = [repo["language"] for repo in last_5_repositories]
