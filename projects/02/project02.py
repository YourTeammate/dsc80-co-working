import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_san(infp, outfp):
    """
    get_san takes in a filepath containing all flights and a filepath where
    filtered dataset #1 is written (that is, all flights arriving or departing
    from San Diego International Airport in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'santest.tmp')
    >>> get_san(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (53, 31)
    >>> os.remove(outfp)
    """

    L = pd.read_csv(infp, chunksize=1000)
    df = next(L)
    fil_df = df[(df['ORIGIN_AIRPORT'] == 'SAN') | (df['DESTINATION_AIRPORT'] == 'SAN')]
    fil_df.to_csv(outfp, mode='a', index=False)
    for df in L:
        fil_df = df[(df['ORIGIN_AIRPORT'] == 'SAN') | (df['DESTINATION_AIRPORT'] == 'SAN')]
        fil_df.to_csv(outfp, mode='a', index=False, header=False)

    return None


def get_sw_jb(infp, outfp):
    """
    get_sw_jb takes in a filepath containing all flights and a filepath where
    filtered dataset #2 is written (that is, all flights flown by either
    JetBlue or Southwest Airline in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'jbswtest.tmp')
    >>> get_sw_jb(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (73, 31)
    >>> os.remove(outfp)
    """

    L = pd.read_csv(infp, chunksize=1000)
    df = next(L)
    fil_df = df[(df['AIRLINE'] == 'B6') | (df['AIRLINE'] == 'WN')]
    fil_df.to_csv(outfp, mode='a', index=False)
    for df in L:
        fil_df = df[(df['AIRLINE'] == 'B6') | (df['AIRLINE'] == 'WN')]
        fil_df.to_csv(outfp, mode='a', index=False, header=False)

    return None


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def data_kinds():
    """
    data_kinds outputs a (hard-coded) dictionary of data kinds, keyed by column
    name, with values Q, O, N (for 'Quantitative', 'Ordinal', or 'Nominal').

    :Example:
    >>> out = data_kinds()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'O', 'N', 'Q'}
    True
    """

    return {'YEAR':'O', 'MONTH':'O', 'DAY':'O', 'DAY_OF_WEEK':'O', 'AIRLINE':'N', 'FLIGHT_NUMBER':'N',
       'TAIL_NUMBER':'N', 'ORIGIN_AIRPORT':'N', 'DESTINATION_AIRPORT':'N',
       'SCHEDULED_DEPARTURE':'O', 'DEPARTURE_TIME':'Q', 'DEPARTURE_DELAY':'Q', 'TAXI_OUT':'Q',
       'WHEELS_OFF':'Q', 'SCHEDULED_TIME':'O', 'ELAPSED_TIME':'Q', 'AIR_TIME':'Q', 'DISTANCE':'Q',
       'WHEELS_ON':'Q', 'TAXI_IN':'Q', 'SCHEDULED_ARRIVAL':'O', 'ARRIVAL_TIME':'O',
       'ARRIVAL_DELAY':'Q', 'DIVERTED':'N', 'CANCELLED':'N', 'CANCELLATION_REASON':'N',
       'AIR_SYSTEM_DELAY':'Q', 'SECURITY_DELAY':'Q', 'AIRLINE_DELAY':'Q',
       'LATE_AIRCRAFT_DELAY':'Q', 'WEATHER_DELAY':'Q'}


def data_types():
    """
    data_types outputs a (hard-coded) dictionary of data types, keyed by column
    name, with values str, int, float.

    :Example:
    >>> out = data_types()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'int', 'str', 'float', 'bool'}
    True
    """

    return {'YEAR':'int', 'MONTH':'int', 'DAY':'int', 'DAY_OF_WEEK':'int', 'AIRLINE':'str', 'FLIGHT_NUMBER':'int',
       'TAIL_NUMBER':'str', 'ORIGIN_AIRPORT':'str', 'DESTINATION_AIRPORT':'str',
       'SCHEDULED_DEPARTURE':'int', 'DEPARTURE_TIME':'float', 'DEPARTURE_DELAY':'float', 'TAXI_OUT':'float',
       'WHEELS_OFF':'float', 'SCHEDULED_TIME':'int', 'ELAPSED_TIME':'float', 'AIR_TIME':'float', 'DISTANCE':'int',
       'WHEELS_ON':'float', 'TAXI_IN':'float', 'SCHEDULED_ARRIVAL':'int', 'ARRIVAL_TIME':'float',
       'ARRIVAL_DELAY':'float', 'DIVERTED':'bool', 'CANCELLED':'bool', 'CANCELLATION_REASON':'str',
       'AIR_SYSTEM_DELAY':'float', 'SECURITY_DELAY':'float', 'AIRLINE_DELAY':'float',
       'LATE_AIRCRAFT_DELAY':'float', 'WEATHER_DELAY':'float'}


# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------

def basic_stats(flights):
    """
    basic_stats takes flights and outputs a dataframe that contains statistics
    for flights arriving/departing for SAN.
    That is, the output should have have two rows, indexed by ARRIVING and
    DEPARTING, and have the following columns:

    * number of arriving/departing flights to/from SAN (count).
    * mean flight (arrival) delay of arriving/departing flights to/from SAN
      (mean_delay).
    * median flight (arrival) delay of arriving/departing flights to/from SAN
      (median_delay).
    * the airline code of the airline with the longest flight (arrival) delay
      among all flights arriving/departing to/from SAN (airline).
    * a list of the three months with the greatest number of arriving/departing
      flights to/from SAN, sorted from greatest to least (top_months).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = basic_stats(flights)
    >>> out.index.tolist() == ['ARRIVING', 'DEPARTING']
    True
    >>> cols = ['count', 'mean_delay', 'median_delay', 'airline', 'top_months']
    >>> out.columns.tolist() == cols
    True
    """
    count = pd.Series([flights.groupby('DESTINATION_AIRPORT').count()['YEAR'].loc['SAN'],
                       flights.groupby('ORIGIN_AIRPORT').count()['YEAR'].loc['SAN']]).rename('count')

    mean_delay = pd.Series(
        [flights.groupby('DESTINATION_AIRPORT').aggregate({'ARRIVAL_DELAY': np.mean})['ARRIVAL_DELAY'].loc['SAN'],
         flights.groupby('ORIGIN_AIRPORT').aggregate({'ARRIVAL_DELAY': np.mean})['ARRIVAL_DELAY'].loc[
             'SAN']]).rename('mean_delay')

    median_delay = pd.Series(
        [flights.groupby('DESTINATION_AIRPORT').aggregate({'ARRIVAL_DELAY': np.median})['ARRIVAL_DELAY'].loc['SAN'],
         flights.groupby('ORIGIN_AIRPORT').aggregate({'ARRIVAL_DELAY': np.median})['ARRIVAL_DELAY'].loc[
             'SAN']]).rename('median_delay')

    airline_dep = flights.groupby(['ORIGIN_AIRPORT', 'AIRLINE']).aggregate({'ARRIVAL_DELAY': np.max}). \
        loc['SAN'].sort_values('ARRIVAL_DELAY').index[-1]
    airline_arr = flights.groupby(['DESTINATION_AIRPORT', 'AIRLINE']).aggregate({'ARRIVAL_DELAY': np.max}). \
        loc['SAN'].sort_values('ARRIVAL_DELAY').index[-1]
    airline = pd.Series([airline_arr, airline_dep]).rename('airline')

    months_dep = flights[flights['ORIGIN_AIRPORT'] == 'SAN'].groupby('MONTH').count().sort_values('YEAR',
                 ascending=False).index[:3].tolist()
    months_arr = flights[flights['DESTINATION_AIRPORT'] == 'SAN'].groupby('MONTH').count().sort_values('YEAR',
                 ascending=False).index[:3].tolist()
    top_months = pd.Series([months_dep, months_arr]).rename('top_months')

    stats = pd.concat([count, mean_delay, median_delay, airline, top_months], axis=1)
    stats.index = ['ARRIVING', 'DEPARTING']
    return stats


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def depart_arrive_stats(flights):
    """
    depart_arrive_stats takes in a dataframe like flights and calculates the
    following quantities in a series (with the index in parentheses):
    - The proportion of flights from/to SAN that
      leave late, but arrive early or on-time (late1).
    - The proportion of flights from/to SAN that
      leaves early, or on-time, but arrives late (late2).
    - The proportion of flights from/to SAN that
      both left late and arrived late (late3).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats(flights)
    >>> out.index.tolist() == ['late1', 'late2', 'late3']
    True
    >>> isinstance(out, pd.Series)
    True
    >>> out.max() < 0.30
    True
    """

    late1 = ((flights['DEPARTURE_DELAY'] > 0) & (flights['ARRIVAL_DELAY'] < 0)).mean()
    late2 = ((flights['DEPARTURE_DELAY'] <= 0) & (flights['ARRIVAL_DELAY'] > 0)).mean()
    late3 = ((flights['DEPARTURE_DELAY'] > 0) & (flights['ARRIVAL_DELAY'] > 0)).mean()

    return pd.Series({'late1': late1, 'late2': late2, 'late3': late3})


def depart_arrive_stats_by_month(flights):
    """
    depart_arrive_stats_by_month takes in a dataframe like flights and
    calculates the quantities in depart_arrive_stats, broken down by month

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats_by_month(flights)
    >>> out.columns.tolist() == ['late1', 'late2', 'late3']
    True
    >>> set(out.index) <= set(range(1, 13))
    True
    """

    return flights.groupby('MONTH').apply(depart_arrive_stats)


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def cnts_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, how many flights were there (in 2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = cnts_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    >>> (out >= 0).all().all()
    True
    """

    counts = flights.groupby(['DAY_OF_WEEK', 'AIRLINE'], as_index=False)['FLIGHT_NUMBER'].count()

    return counts.pivot(index='DAY_OF_WEEK', columns='AIRLINE', values='FLIGHT_NUMBER')



def mean_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, what is the average ARRIVAL_DELAY (in
    2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = mean_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    """

    avgs = flights.groupby(['DAY_OF_WEEK', 'AIRLINE'], as_index=False)['ARRIVAL_DELAY'].mean()

    return avgs.pivot(index='DAY_OF_WEEK', columns='AIRLINE', values='ARRIVAL_DELAY')


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def predict_null_arrival_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the ARRIVAL_DELAY is null and otherwise False.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `ARRIVAL_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('ARRIVAL_DELAY', axis=1).apply(predict_null_arrival_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """


    return np.isnan(row['ELAPSED_TIME'])


def predict_null_airline_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the AIRLINE_DELAY is null and otherwise False. Since the
    function doesn't depend on AIRLINE_DELAY, it should work a row even if that
    index is dropped.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `AIRLINE_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('AIRLINE_DELAY', axis=1).apply(predict_null_airline_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """

    return np.isnan(row['AIR_SYSTEM_DELAY'])


# ---------------------------------------------------------------------
# Question #7
# ---------------------------------------------------------------------

def perm4missing(flights, col, N):
    """
    perm4missing takes in flights, a column col, and a number N and returns the
    p-value of the test (using N simulations) that determines if
    DEPARTURE_DELAY is MAR dependent on col.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = perm4missing(flights, 'AIRLINE', 100)
    >>> 0 <= out <= 1
    True
    """

    obs_pt = (
        flights
        .assign(is_null = flights.DEPARTURE_DELAY.isnull())
        .pivot_table(index = 'is_null', columns = col, aggfunc = 'size')
        .apply(lambda x:x / x.sum(), axis=1)
    )
    obs_tvd = obs_pt.fillna(0).diff().iloc[-1].abs().sum() / 2

    shuffled = flights.assign(is_null = flights.DEPARTURE_DELAY.isnull())
    shuffle_col = shuffled['is_null'].values
    tvds = []

    for _ in range(N):
        shuffled_col = np.random.permutation(shuffle_col)
        shuffled['is_null'] = shuffled_col

        pt = (
            shuffled
            .pivot_table(index = 'is_null', columns = col, aggfunc = 'size')
            .apply(lambda x:x / x.sum(), axis=1)
        )

        tvds.append(pt.fillna(0).diff().iloc[-1].abs().sum() / 2)

    return np.mean(tvds >= obs_tvd)


def dependent_cols():
    """
    dependent_cols gives a list of columns on which DEPARTURE_DELAY is MAR
    dependent on.

    :Example:
    >>> out = dependent_cols()
    >>> isinstance(out, list)
    True
    >>> cols = 'YEAR DAY_OF_WEEK AIRLINE DIVERTED CANCELLATION_REASON'.split()
    >>> set(out) <= set(cols)
    True
    """

    return ['DAY_OF_WEEK', 'AIRLINE']


def missing_types():
    """
    missing_types returns a Series
    - indexed by the following columns of flights:
    CANCELLED, CANCELLATION_REASON, TAIL_NUMBER, ARRIVAL_TIME.
    - The values contain the most-likely missingness type of each column.
    - The unique values of this Series should be MD, MCAR, MAR, MNAR, NaN.

    :param:
    :returns: A series with index and values as described above.

    :Example:
    >>> out = missing_types()
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) - set(['MD', 'MCAR', 'MAR', 'NMAR', np.NaN]) == set()
    True
    """

    ans = {'CANCELLED': np.NaN, 'CANCELLATION_REASON': 'MD', 'TAIL_NUMBER': 'MCAR', 'ARRIVAL_TIME': 'MAR'}

    return pd.Series(ans)


# ---------------------------------------------------------------------
# Question #8
# ---------------------------------------------------------------------

def prop_delayed_by_airline(jb_sw):
    """
    prop_delayed_by_airline takes in a dataframe like jb_sw and returns a
    DataFrame indexed by airline that contains the proportion of each airline's
    flights that are delayed.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> (out >= 0).all().all() and (out <= 1).all().all()
    True
    >>> len(out.columns) == 1
    True
    """
    airports_set = set(['ABQ', 'BDL', 'BUR', 'DCA', 'MSY', 'PBI', 'PHX', 'RNO', 'SJC', 'SLC'])
    jb_sw = jb_sw[jb_sw['ORIGIN_AIRPORT'].apply(lambda x: x in airports_set)]

    jb_sw_p = jb_sw.assign(delayed=((jb_sw['DEPARTURE_DELAY'] > 0) & (jb_sw['CANCELLED'] == 0)))

    return jb_sw_p.groupby('AIRLINE')['delayed'].mean().to_frame()


def prop_delayed_by_airline_airport(jb_sw):
    """
    prop_delayed_by_airline_airport that takes in a dataframe like jb_sw and
    returns a DataFrame, with columns given by airports, indexed by airline,
    that contains the proportion of each airline's flights that are delayed at
    each airport.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline_airport(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> ((out >= 0) | (out <= 1) | (out.isnull())).all().all()
    True
    >>> len(out.columns) == 6
    True
    """

    airports_set = set(['ABQ', 'BDL', 'BUR', 'DCA', 'MSY', 'PBI', 'PHX', 'RNO', 'SJC', 'SLC'])
    jb_sw = jb_sw[jb_sw['ORIGIN_AIRPORT'].apply(lambda x: x in airports_set)]

    jb_sw_p = jb_sw.assign(delayed=((jb_sw['DEPARTURE_DELAY'] > 0) & (jb_sw['CANCELLED'] == 0)))

    return jb_sw_p.pivot_table(index = 'AIRLINE', columns = 'ORIGIN_AIRPORT', values = 'delayed', aggfunc = 'mean')


# ---------------------------------------------------------------------
# Question #9
# ---------------------------------------------------------------------

def verify_simpson(df, group1, group2, occur):
    """
    verify_simpson verifies whether a dataset displays Simpson's Paradox.

    :param df: a dataframe
    :param group1: the first group being aggregated
    :param group2: the second group being aggregated
    :param occur: a column of df with values {0,1}, denoting
    if an event occurred.
    :returns: a boolean. True if simpson's paradox is present,
    otherwise False.

    :Example:
    >>> df = pd.DataFrame([[4,2,1], [1,2,0], [1,4,0], [4,4,1]], columns=[1,2,3])
    >>> verify_simpson(df, 1, 2, 3) in [True, False]
    True
    >>> verify_simpson(df, 1, 2, 3)
    False
    """

    rate1 = df.groupby(group1)[occur].mean()
    rate1 = rate1.diff().iloc[-1]

    rate2 = df.pivot_table(index = group1, columns = group2, values = occur, aggfunc = 'mean')
    rate2 = rate2.diff().iloc[-1]

    if rate1 > 0:
        return (rate2 < 0).all()
    else:
        return (rate2 > 0).all()


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def search_simpsons(jb_sw, N):
    """
    search_simpsons takes in the jb_sw dataset and a number N, and returns a
    list of N airports for which the proportion of flight delays between
    JetBlue and Southwest satisfies Simpson's Paradox.

    Only consider airports that have '3 letter codes',
    Only consider airports that have at least one JetBlue and Southwest flight.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=1000)
    >>> pair = search_simpsons(jb_sw, 2)
    >>> len(pair) == 2
    True
    """

    return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_san', 'get_sw_jb'],
    'q02': ['data_kinds', 'data_types'],
    'q03': ['basic_stats'],
    'q04': ['depart_arrive_stats', 'depart_arrive_stats_by_month'],
    'q05': ['cnts_by_airline_dow', 'mean_by_airline_dow'],
    'q06': ['predict_null_arrival_delay', 'predict_null_airline_delay'],
    'q07': ['perm4missing', 'dependent_cols', 'missing_types'],
    'q08': ['prop_delayed_by_airline', 'prop_delayed_by_airline_airport'],
    'q09': ['verify_simpson'],
    'q10': ['search_simpsons']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """

    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
