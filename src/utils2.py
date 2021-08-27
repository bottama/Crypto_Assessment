""" Self made function to solve task 2 """


# get historical data
def historical_data(market, days, interval):

    # import modules
    import requests
    import json
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    import math

    # end date and start date
    end_date = datetime.now().replace(microsecond=0, second=0, minute=0)
    start_date = end_date - timedelta(days=days)

    iterations = math.ceil(days / (1500 / 24 * interval))
    flag = False
    for i in range(0, iterations):

        # start, end timestamps
        end = start_date + timedelta(hours=1500)

        # for next iteration
        start_date = end

        # int
        start = int((end-timedelta(hours=1500)).replace(tzinfo=timezone.utc).timestamp())
        end = int(end.replace(tzinfo=timezone.utc).timestamp())

        # API request
        request = requests.get('https://ftx.com/api/markets/' + market + '/candles?resolution=3600&start_time=' +
                               str(start) + '&end_time=' + str(end))
        result = json.loads(request.text)

        if not flag:  # first iteration only
            df = pd.json_normalize(result['result'])
            flag = True
        else:  # then append
            df2 = pd.json_normalize(result['result'])
            df = pd.concat([df, df2])
            del df2

        # next iteration
        i += 1

    # adjust df columns and names
    df.set_index('startTime', inplace=True)
    df.columns = df.columns +  '_' + market

    print(market + ' ' + str(interval) + ' hour data for the last ' + str(days) + ' days: DONE')
    return df


# Task 2.b. analysis
def historical_analysis(df, markets, local_folder, interval, rolling_days, days):

    # import modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # folder path
    folder_path = os.path.abspath(os.getcwd())
    folder_path_table = folder_path.replace(local_folder, 'reports/task2/tables/')
    folder_path_plot = folder_path_table.replace('tables', 'images')

    # evaluate returns
    flag = False
    for market in markets:
        if not flag:
            ret = df['close_' + market] / df['close_' + market].shift(1) - 1
            ret = pd.DataFrame(ret)
            flag = True
        else:
            ret2 = df['close_' + market] / df['close_' + market].shift(1) - 1
            ret2 = pd.DataFrame(ret2)
            ret = ret.join(ret2)
            del ret2

    # rename columns
    ret.columns = markets

    # summary statistics
    summary = ret.describe()
    file_path = folder_path_table + 'summary_statistics_ret.csv'
    summary.to_csv(file_path)

    # histogram
    file_path_plot = folder_path_plot + 'histogram_ret.png'
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s='Returns distribution' +
               ' \n Last ' + str(days) + ' days, ' + str(interval) + ' hour interval ',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    for market in markets:
        plt.hist(ret[market], alpha=0.5, label=market)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.legend(markets, bbox_to_anchor=(.5, 0.03), loc="lower center",
              bbox_transform=fig.transFigure, ncol=len(markets), frameon=False)
    plt.savefig(file_path_plot, dpi=160)  # save fig
    plt.close()

    # rolling correlations
    rolling_correlation = ret[markets[0]].rolling(interval*24*rolling_days).corr(ret[markets[1]])

    # cumulative performance
    cum_ret = (1 + ret).cumprod()

    # realized volatility
    ret.index = pd.to_datetime(ret.index)
    realized_volatility = (ret ** 2).resample('D').sum()

    # realized volatility plot
    file_path_plot = folder_path_plot + 'realized_volatility.png'
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s='Realized Volatility' +
               ' \n Last ' + str(days) + ' days, ' + str(interval) + ' hour interval returns squared',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    ax.plot(realized_volatility)
    ax.margins(x=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(markets, bbox_to_anchor=(.5, 0.03), loc="lower center",
              bbox_transform=fig.transFigure, ncol=len(markets), frameon=False)
    fig.text(0.06, 0.5, 'Realized Volatility', ha='center', va='center', rotation='vertical')
    plt.savefig(file_path_plot, dpi=160)  # save fig
    plt.close()

    # performance & rolling correlations
    file_path_plot = folder_path_plot + 'performance_and_rolling_correlation.png'
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    fig.text(s='Cumulative Returns' +
               ' \n Last ' + str(days) + ' days',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    ax[0].plot(cum_ret)
    ax[1].plot(rolling_correlation, color='black')
    ax[0].margins(x=0)
    ax[1].margins(x=0)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    x_ticks = [cum_ret.index[0], cum_ret.index[round(len(cum_ret.index)/2)], cum_ret.index[-1]]
    ax[1].set_xticks(x_ticks)
    fig.text(0.06, 0.7, 'Performance', ha='center', va='center', rotation='vertical')
    fig.text(0.06, 0.3, str(rolling_days) + 'D rolling correlation', ha='center', va='center', rotation='vertical')
    fig.legend(markets, bbox_to_anchor=(.5, 0.03), loc="lower center",
              bbox_transform=fig.transFigure, ncol=len(markets), frameon=False)
    plt.savefig(file_path_plot, dpi=160)  # save fig
    plt.close()

    print('Task 2.b: DONE')
    return ret, rolling_correlation, realized_volatility


# Task 2.c. Cointegration test
def cointegration_test(df, local_folder):

    # import modules
    import statsmodels.tsa.stattools as ts
    import pandas as pd
    import os

    # folder path
    folder_path = os.path.abspath(os.getcwd())
    folder_path_table = folder_path.replace(local_folder, 'reports/task2/tables/')

    x = df['close_ETH/USD']
    y = df['close_BTC/USD']
    coin_result = ts.coint(x, y)

    cointegration_df = pd.DataFrame(
        {
            "t-test": [coin_result[0]],
            "p-value": [coin_result[1]],
            "crit_value_1%": [coin_result[2][0]],
            "crit_value_5%": [coin_result[2][1]],
            "crit_value_10%": [coin_result[2][2]]
    })

    cointegration_df.to_csv(folder_path_table + 'cointegration_test.csv')

    print('task 2.c.: DONE')
    return cointegration_df


# task 2
def task2(markets, days, interval, rolling_days,local_folder):

    # task 2.a
    flag = False
    if len(markets) > 1:
        for market in markets:
            if not flag:
                df = historical_data(market=market, days=days, interval=interval)
                flag = True
            else:
                df2 = historical_data(market=market, days=days, interval=interval)
                df = df.join(df2)
                del df2
    else:
        df = historical_data(market=markets, days=days, interval=interval)
    print('datasets merge: DONE')
    print('Task 2.a: DONE')

    # task 2.b
    analytics = historical_analysis(df=df, markets=markets, local_folder=local_folder,
                                    interval=interval, rolling_days=rolling_days, days=days)

    # task 2.c
    cointegration_df = cointegration_test(df=df, local_folder=local_folder)

    return analytics, cointegration_df
