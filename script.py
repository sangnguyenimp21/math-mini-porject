import requests
import csv
import time
import sys, getopt
import datetime
import os
from dotenv import load_dotenv

from stocks import HNX_STOCK_LIST

load_dotenv()

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
# https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=~date:gte:2021-01-01~date:lte:2021-06-30&size=5&page=2
BASE_URL = 'https://finfo-api.vndirect.com.vn/'
STOCK_PRICES_URL = BASE_URL+'v4/stock_prices'

SIZE = 20
FORMAT = "%Y-%m-%d"
BASE_START_DATE = '2021-01-01' #format yyyy-MM-dd
BASE_END_DATE = '2021-01-04'

FIELDS = [
    'code', 'date', 'time', 'floor',
    'type', 'basicPrice', 'ceilingPrice', 'floorPrice',
    'open', 'high', 'low', 'close',
    'average', 'adOpen', 'adHigh', 'adLow',
    'adClose', 'adAverage', 'nmVolume', 'nmValue',
    'ptVolume', 'ptValue', 'change', 'adChange', 'pctChange',
]

def init(start_date = BASE_START_DATE, end_date = BASE_END_DATE, size = SIZE):
    url =  get_url(start_date, end_date, size)
    response = requests.get(url, headers=HEADERS)
    json_content = response.json()
    total_elements = json_content['totalElements']
    total_pages = json_content['totalPages']

    return total_pages, total_elements, size

def get_url(start_date = BASE_START_DATE, end_date = BASE_END_DATE, size = SIZE, page = None):
    url = STOCK_PRICES_URL+'?'
    url += 'sort=date'
    url += '&q=~date:gte:'+start_date+'~date:lte:'+end_date
    url += '&size='+str(size)
    if page is not None:
        url += '&page='+str(page)
    return url

def get_data(start_date=BASE_START_DATE, end_date=BASE_END_DATE, size=SIZE, page=1):
    url = get_url(start_date, end_date, size, page)
    response = requests.get(url, headers=HEADERS)
    json_content = response.json()
    data = json_content['data']
    return data

def init_csv_file(start_date = BASE_START_DATE, end_date =  BASE_END_DATE):
    ts = time.time()
    ts = int(ts)
    filename = 'data_stock_'+start_date+'_'+end_date+'_'+(str(ts))+'.csv'
    if(os.getenv('VIEW_ONLY') not in ['True', 'true', '1']):
        with open(filename, mode='w',  newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            writer.writeheader()

    return filename

def write_to_csv(filename, ele, floor = 'HNX', code='', is_all_code = True):
    if(os.getenv('VIEW_ONLY') not in ['True', 'true', '1']):
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if (ele['floor'] == floor):
                if is_all_code is True:
                    writer.writerow(ele)
                else:
                    if code != '':
                        if ele['code'] == code:
                            writer.writerow(ele)
    else:
        if (ele['floor'] == floor):
            if is_all_code is True:
                print(ele)
            else:
                if code != '':
                    if ele['code'] == code:
                        print(ele)

def run(argv):
    start_date = BASE_START_DATE
    start_ts = int(datetime.datetime.strptime(start_date, FORMAT).timestamp())
    end_date = BASE_END_DATE
    end_ts = int(datetime.datetime.strptime(end_date, FORMAT).timestamp())
    size = SIZE
    floor = 'HNX'
    code = ''
    is_all_code = True

    options = "h:b:e:s:f:c"
    long_options = ["Help","Begin", "End", "Size", "Floor", "Code"]

    try:
      opts, args = getopt.getopt(argv, options, long_options)
    except getopt.GetoptError:
        print('script.py -b <Start date> -e <End date> -s <Size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--Help"):
            print('script.py -b <Start date> -e <End date> -s <Size>')
            sys.exit()
        elif opt in ("-b", "--Begin"):
            start_date = arg
            try:
                start_ts = int(datetime.datetime.strptime(start_date, FORMAT).timestamp())
            except ValueError:
                print("Incorrect start date format. It should be YYYY-MM-DD")
                exit(0)
        elif opt in ("-e", "--End"):
            end_date = arg
            try:
                end_ts = int(datetime.datetime.strptime(end_date, FORMAT).timestamp())
            except ValueError:
                print("Incorrect end date format. It should be YYYY-MM-DD")
                exit(0)
        elif opt in ("-s", "--Size"):
            size = arg
            try:
                size = int(size)
            except ValueError:
                print("Size must be integer")
                exit(0)

            if (int(size) <= 0):
                print('Size is not valid')
                exit(0)
        elif opt in ("-f", "--Floor"):
            floor = arg
        elif opt in ("-c", "--Code"):
            code = arg
            is_all_code = False

    if (end_ts < start_ts):
        print('End Date must greater than Start Date')
        exit(0)


    file_name = init_csv_file(start_date, end_date)
    total_pages, total_elements, size =  init(start_date = start_date, end_date = end_date, size = size)

    assert (int(total_elements) != 0 and int(total_pages) != 0), 'No Data Retrieved'
    
    for page in range(1, total_pages+1):
        data = get_data(start_date, end_date, size, page)
        for ele in data:
            write_to_csv(file_name, ele, floor, code, is_all_code)

    return file_name

if __name__ == '__main__':
    file_name = run(sys.argv[1:])
