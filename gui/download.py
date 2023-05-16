import requests

headers = {"Accept-Encoding": "gzip",
           'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
           }

# -------------------------------------------------
# get_content
# -------------------------------------------------
def get_content(url):
    
    response = requests.get(url, headers=headers).json()
    return response

if __name__ == '__main__':

    #url = 'https://api.nasdaq.com/api/quote/TSLA/realtime-trades?&limit=100&offset=0&fromTime=00:00'
    url = 'https://api.nasdaq.com/api/quote/TSLA/realtime-trades?&limit=10&offset=0&fromTime=09:30'
    data = get_content(url)
    
    if data is not None:

        data = data['data']
        totalRecords = data['totalRecords']
        offset = data['offset']
        limit = data['limit']

        rows = data['rows']
        #print (rows)
        print ('LEN', len(rows))

        print ('totalRecords', totalRecords)

        print (rows[0])
        print (rows[-1:])

    else:
        print ("ERROR")