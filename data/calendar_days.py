# data/calendar_days.py
from datetime import datetime, timedelta, date

''' Календарь праздников '''

# Рождественские праздники пока списком создал
christmas_holid = [date(year=2025, month=12, day=31) + timedelta(days=i) for i in range(12)]

# Праздники рандомные
HOLIDAYS = set(
    [
    datetime(2025, 11, 3).date(),
    datetime(2025, 11, 4).date(),
    datetime(2025, 6, 12).date(),
    datetime(2025, 6, 13).date(),
    datetime(2025, 5, 1).date(),
    datetime(2025, 5, 2).date(),
    datetime(2025, 5, 3).date(),
    datetime(2025, 5, 4).date(),
    datetime(2025, 5, 8).date(),
    datetime(2025, 5, 9).date(),
    datetime(2025, 5, 10).date(),
    datetime(2025, 5, 11).date(),
    *christmas_holid
])
