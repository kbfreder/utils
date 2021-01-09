import datetime as dt
from dateutil.relativedelta import relativedelta as rd
from calendar import monthrange as mr


def date_months_prior(n):
        '''Return datetime object x months prior from today
        rounded to first day of month'''
        t = dt.datetime.today()
        exact_dt = t - rd(months=n)
        return dt.datetime(exact_dt.year, exact_dt.month, 1)


def last_months_last_day():
        '''Return datetime object for last day of previous month
        
                Ex: today = 2019-06-15
                
                last_months_last_day()
                >>> Timestamp (2019, 5, 31)
        '''
        t = dt.datetime.today()
        m = t.month
        y = t.year
        if m == 1:
                y -= 1
                m = 12
        else:
                m -= 1
        
        d = mr(y, m)[1]

        return dt.datetime(y, m, d)
