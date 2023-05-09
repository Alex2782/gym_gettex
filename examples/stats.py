from utils import *
from tqdm import tqdm

max_data = 256
window_size = 30
min_df_len = 1000

date = None
#date = '2023-04-13'

isin_list = [] #if empty -> load from '/Users/alex/Develop/gettex/finanzen.net.pickle'
#isin_list += ["DE0007236101", "DE0008232125", "US83406F1021", "FI0009000681"]
#isin_list += ["GB00BYQ0JC66"]
isin_list += ["US88160R1014"] #tesla
#isin_list += ['US83304A1060','US12468P1049','US88160R1014','US19260Q1076','US62914V1061','US86771W1053']

#isin_list += ["US09075V1026"] # Biontech
#isin_list += ["US0846707026"] # Berkshire Hathaway Inc.
#isin_list += ["US9024941034"] # Tyson Foods Inc.



# https://mein.finanzen-zero.net/assets/searchdata/downloadable-instruments.csv
# create pickle file: https://github.com/Alex2782/gettex-import/blob/main/finanzen_net.py
if isin_list is None or len(isin_list) == 0:
    isin_list = get_finanzen_stock_isin_list()
    np.random.shuffle(isin_list)

batch_list = np.array_split(isin_list, len(isin_list)/max_data + 1)
len_batch = len(batch_list)

line = '-' * 170
grey_color = '\033[2m'
red_color = '\033[31m'
green_color = '\033[32m'
normal_color = '\033[0m'
I = f'{grey_color}|{normal_color}'

def num_color(val):
    color = ''
    if val < 0: color = f'{red_color}'
    elif val > 0: color = f'{green_color}'

    return f'{color}{val:>6.1f}{normal_color}'

stats = {}
isin_counter = {}


_HEADER_OUT =   f'{"isin":^8} {"":^6} {"date":^10} {I} {"T.US":^5} {I} {"T.DE":^5} {I} '\
                f'{"prev.D.C":^9} {"last_close":^9} '\
                f'{"diff.PRE":^9} {I} '\
                f'{"diff.POST":^9} {"diff_high":^9} {"diff_low":^9} ({"H / L":^5}){I} '\
                f'{"open":^11} {"high":^10} {"low":^7} {"close":^9} {"vola_profit":^9}'

_OUT = []
b_header_out = False

tbar = tqdm(range(0, len_batch))
for i in tbar:

    #print ('BATCH:', i + 1, ' / ', len_batch)
    #print ('-' * 50)

    batch_isin = batch_list[i]

    batch_isin, df_list, skip_counter = load_data(window_size, batch_isin, date, None, min_df_len)
    #print (f'{skip_counter} files skipped')

    for df_dict in df_list:
        isin = df_dict['isin']
        df = df_dict['df']

        tm = df['Date'].astype(int).astype(str) + ' ' + df['HH'].astype(int).astype(str) + ':' + df['MM'].astype(int).astype(str)
        df['DateTime'] = pd.to_datetime(tm).dt.tz_localize('UTC')
        df['DateTime_DE'] = df['DateTime'].dt.tz_convert('Europe/Berlin')
        df['DateTime_US'] = df['DateTime'].dt.tz_convert('US/Eastern')


        _date_prev = 0
        _last_close = 0
        _prev_day_close = 0
        b_header_out = False

        for index, row in df.iterrows():
            _date = row['Date']
            _date_f = row['DateTime'].strftime('%Y-%m-%d')
            _time_us = row['DateTime_US'].strftime('%H:%M')
            _time_de = row['DateTime_DE'].strftime('%H:%M')
            _weekday = row['DateTime'].strftime('%A')[:2]
            _weekday_int = row['DateTime'].strftime('%w')

            HH = row['HH']
            MM = row['MM']
            _open = row['Open']
            _high = row['High']
            _low = row['Low']
            _close = row['Close']
            tm = row['DateTime']
            vola_profit = row['vola_profit']

            #if _date_f < '2023-03-12': continue  # ignore US - Wintertime 
            
            # (last) 4 Weeks 
            if _date_f < '2023-04-08': continue  
            if _date_f > '2023-06-01': continue 

            if _open < 2: break

            if _date_prev > 0 and _date_prev != _date:
                _prev_day_close = _last_close
                pass
            
            _check_time = '09:45'
            _check_time_type = _time_us

            #_check_time = '08:15'
            #_check_time_type = _time_de

            if _prev_day_close > 0 and _check_time_type == _check_time:
            
                diff_prev_day = (_last_close - _prev_day_close) / _prev_day_close * 100
                diff_close = (_close - _last_close) / _last_close * 100
                diff_high = (_high - _last_close) / _last_close * 100
                diff_low = (_low - _last_close) / _last_close * 100
                diff_hl = (_high - _low) / _last_close * 100

                if vola_profit > 1:
                    #key = f'{int(HH):02d}:{int(MM):02d}'
                    key = f'{_weekday_int} - {_weekday} {_check_time}'
                    if (diff_prev_day < 0 and diff_close > 0) or (diff_prev_day > 0 and diff_close < 0): key += ' <> '
                    elif (diff_prev_day > 0 and diff_close > 0) or (diff_prev_day < 0 and diff_close < 0): key += ' == '
                    else: key += '    '

                    if stats.get(key) is None: stats[key] = {'counter': 0, 'H':[], 'L':[], 'C':[], 'HL':[]}
                    stats[key]['counter'] += 1
                    stats[key]['H'].append(diff_high)
                    stats[key]['L'].append(diff_low)
                    stats[key]['C'].append(diff_close)
                    stats[key]['HL'].append(diff_hl)

                    if isin_counter.get(isin) is None: isin_counter[isin] = 0
                    isin_counter[isin] += 1


                    out = f'{isin} {_weekday} {_date_f} {I} {_time_us} {I} {_time_de} {I} {_prev_day_close:>9.3f} {_last_close:>9.3f} '\
                          f'  {num_color(diff_prev_day)} % {I} '\
                          f'{num_color(diff_close)} % {diff_high:>7.1f} % {diff_low:>7.1f} %  ({diff_hl:>3.1f} %){I} '\
                          f'{_open:>9.3f} {_high:>9.3f} {_low:>9.3f} {_close:>9.3f} {vola_profit:>9.3f}'
                    
                    if not b_header_out:
                        _OUT.append(line)
                        _OUT.append(_HEADER_OUT)
                        _OUT.append(line)
                        b_header_out = True

                    _OUT.append(out)
                pass

            _date_prev = _date
            _last_close = _close
            pass

tbar.close()

for out in _OUT:
    print(out)

# stats
print (f'{"KEY":^16} {"counter":>9} {I} {"H":>9} {"L":>9} {"C":>9} {"HL":>9}')
for key in sorted(stats.keys()):
    s = stats[key]
    s["H"] = np.median(s["H"])
    s["L"] = np.median(s["L"])
    s["C"] = np.median(s["C"])
    s["HL"] = np.median(s["HL"])

    print(f'{key:<10} {s["counter"]:>9d} {I} {s["H"]:>9.2f} {s["L"]:>9.2f} {s["C"]:>9.2f} {s["HL"]:>9.2f}')

print(line)
# isin_counter
sorted_dict = dict(sorted(isin_counter.items(), key=lambda x: x[1], reverse=True))

for key in sorted_dict:
    if sorted_dict[key] > 10:
        print(key, ':', sorted_dict[key])

