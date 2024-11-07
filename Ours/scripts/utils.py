import time
from datetime import datetime
from time import mktime
import pytz  # 时区计算


"""
    处理时间戳
    转换的最小单位是毫秒
"""


def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    date, ns = date.split('.')

    timeArray = time.strptime(date, '%Y-%m-%dT%H:%M:%S')
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    timeStamp += int(ns.split('Z')[0])
    return timeStamp


def datetime_to_timestamp_US(date):
    """
    :param date: str   e.g. 2013-10-10T23:40:00.974-04:00
    :return: nano timestamp
    """
    date = date.replace('-04:00', '')
    if '.' in date:
        date, ms = date.split('.')
    else:
        ms = 0
    tz = pytz.timezone('Etc/GMT+4')
    timeArray = time.strptime(date, "%Y-%m-%dT%H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000 + int(ms)
    return int(timeStamp)


def timestamp_to_datetime_US(ns):
    """
    :param date: nano_timestamps:str   format: 1533289579012(000000)
    :return: nano timestamp e.g. 2018-04-09 11:59:39.12
    """
    tz = pytz.timezone('US/Eastern')
    # ms = ns % 1000
    ms = str(ns)[-3:]
    ns /= 1000
    dt = pytz.datetime.datetime.fromtimestamp(int(ns), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(ms)
    #     s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s


if __name__ == '__main__':
    # print(datetime_to_ns_time())
    # print(datetime_to_timestamp_US("2019-09-23T09:42:55.974-04:00"))
    print(timestamp_to_datetime_US(1523289468989))