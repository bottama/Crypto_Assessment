""" Self made function to solve task 1"""


# 1.a. merge dataset
def merge_dataset(local_folder, data_folder, df1='data_nonfiltrata_a.csv', df2='data_nonfiltrata_b.csv'):

    # import modules
    import pandas as pd
    import os

    # folder path
    folder_path = os.path.abspath(os.getcwd())
    folder_path = folder_path.replace(local_folder, data_folder)

    # merge data
    elements = [df1, df2]

    flag = False
    for element in elements:
        file_path = folder_path + '/' + element

        if not flag:
            df = pd.read_csv(file_path, index_col=0)
            flag = True
        else:
            df2 = pd.read_csv(file_path, index_col=0)
            df = df.merge(df2)
            df.sort_values('timestamp')

    print('Task 1.a: Merge both datasets using the common variable and sort the data by timestamp: DONE')
    return df


# 1.b. clean data
def clean_data(merged_df):

    # drop NA
    len_init = len(merged_df)
    df = merged_df.dropna(axis=0)[:-1]
    len_final = len(df)

    print('Task 1.b: clean data: DONE')
    print('final length: ' + str(len_final) + ' , row deleted: ' + str(len_init - len_final))

    return df


# 1.c. Timestamp CET
def timestamp_CET(df):

    # import modules
    import pandas as pd

    # add CET timestamp data
    df['timestamp_CET'] = pd.to_datetime(df['timestamp'], unit='us')
    df['timestamp_CET'] = df['timestamp_CET'].astype("datetime64[ns, CET]")

    print('Task 1.c: add Timestamp CET data: DONE')
    return df


# 1.d. plot
def price_amount_plot(df, local_folder):

    # import modules
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    import os

    # folder path
    folder_path = os.path.abspath(os.getcwd())
    folder_path = folder_path.replace(local_folder, 'reports/task1/images/')

    # file name and path
    file_name = df.symbol.unique()[0]
    file_path = folder_path + file_name + '.png'

    # plot
    plt.rcParams['font.family'] = 'serif'  # set font family: serif

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    fig.text(s=df.symbol.unique()[0] + ' \n Price and Volume dynamics',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    ax[0].plot(df['timestamp_CET'], df['price'])
    ax[1].plot(df['timestamp_CET'], df['amount'])
    fig.text(0.06, 0.7, df.symbol.unique()[0], ha='center', va='center', rotation='vertical')
    fig.text(0.06, 0.3, 'Volume', ha='center', va='center', rotation='vertical')
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].margins(x=0)
    ax[1].margins(x=0)
    ax[1].margins(y=0)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.savefig(file_path, dpi=160)  # save fig
    plt.close()

    return print('Task 1.d: Plot Price and Volume: DONE')


# 1.e. Clean the outliers in variable amount
def remove_outliers(df, feature, local_folder):

    # import modules
    import pandas as pd
    from scipy import stats
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

    # folder path
    folder_path = os.path.abspath(os.getcwd())
    folder_path_table = folder_path.replace(local_folder, 'reports/task1/tables/')
    folder_path_plot = folder_path_table.replace('tables', 'images')

    # file name and path
    file_name = df.symbol.unique()[0] + '_' + feature
    file_path_table = folder_path_table + file_name + '.csv'
    file_path_plot = folder_path_plot + file_name + '.png'

    # z-scores
    z_scores = stats.zscore(df[feature])

    # filter obs < 3 std
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    new_feature = df[feature][filtered_entries]

    # summary statistics
    summary_init = df[feature].describe()
    summary_final = new_feature.describe()
    summary_stat = pd.concat([summary_init, summary_final], axis=1)
    summary_stat.columns = ['with_outliers', 'no_outliers']

    # create .csv summary statistics
    summary_stat.to_csv(file_path_table)
    print('Summary statistics ' + feature + ': DONE')

    # box plot
    df_box_plot = [df[feature], new_feature]
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=df.symbol.unique()[0] +
               ' Volume Boxplot \n Outlier: value greater than 3 standard deviations from the mean',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    bp = ax.boxplot(df_box_plot, patch_artist=True)
    for box in bp['boxes']:
        box.set(color='#7570b3', linewidth=2)
        box.set(facecolor='#1b9e77')
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    ax.set_xticklabels(['With Outliers', 'Without Outliers'])
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    fig.text(0.06, 0.5, 'Volume', ha='center', va='center', rotation='vertical')
    plt.savefig(file_path_plot, dpi=80)  # save fig
    plt.close()

    # box plot without outliers alone
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=df.symbol.unique()[0] +
               ' Volume Boxplot \n Outlier: value greater than 3 standard deviations from the mean',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    bp = ax.boxplot(new_feature, patch_artist=True)
    for box in bp['boxes']:
        box.set(color='#7570b3', linewidth=2)
        box.set(facecolor='#1b9e77')
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    ax.set_xticklabels(['Without Outliers'])
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    fig.text(0.06, 0.5, 'Volume', ha='center', va='center', rotation='vertical')
    plt.savefig(file_path_plot.replace('.png', '_no_outliers.png'), dpi=80)  # save fig
    plt.close()

    print('Boxplot ' + feature + ': DONE')
    return print('Task 1.e: Clean the outliers in the variable ' + feature + ': DONE')


# 1.f. 100 point simple moving average
def sma(df, feature, N, local_folder):

    # import modules
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # folder path
    folder_path = os.path.abspath(os.getcwd())
    folder_path_table = folder_path.replace(local_folder, 'reports/task1/tables/')
    folder_path_plot = folder_path_table.replace('tables', 'images')

    # file name and path
    file_name = df.symbol.unique()[0] + '_' + feature + '_sma' + str(N)
    file_path_table = folder_path_table + file_name + '.csv'
    file_path_plot = folder_path_plot + file_name + '.png'

    # Simple Moving Average (SMA)
    x = df[feature].values
    cumsum = np.cumsum(np.insert(x, 0, 0))
    sma = pd.Series((cumsum[N:] - cumsum[:-N]) / float(N))

    # append values
    value = pd.Series(np.zeros(N-1))
    value = value.replace(0, np.nan)
    sma = value.append(sma)
    df['sma_' + str(N)] = sma.values

    # comparing summary statistics
    summary_sma = sma.dropna().describe()
    summary_feature = df[feature].describe()
    summary_stat = pd.concat([summary_feature, summary_sma], axis=1)
    summary_stat.columns = [feature, 'SMA_' + str(N)]
    summary_stat.to_csv(file_path_table)

    # plot
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=df.symbol.unique()[0] + ' \n' + feature + ' vs SMA ' + str(N),
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    ax.plot(df['timestamp_CET'], df[feature])
    ax.plot(df['timestamp_CET'], sma)
    fig.text(0.06, 0.5, feature, ha='center', va='center', rotation='vertical')
    ax.margins(x=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend([feature, 'SMA ' + str(N)], bbox_to_anchor=(.5, 0.03),
              loc="lower center", bbox_transform=fig.transFigure, ncol=2, frameon=False)
    plt.savefig(file_path_plot, dpi=160)  # save fig
    plt.close()

    print('Task 1.f: ' + feature + ' and SMA ' + str(N) + ' comparison: DONE')
    return df


# 1.g. 100 point weighted moving average
def wma(df, feature, weight, N, local_folder):

    # import modules
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # folder path
    folder_path = os.path.abspath(os.getcwd())
    folder_path_table = folder_path.replace(local_folder, 'reports/task1/tables/')
    folder_path_plot = folder_path_table.replace('tables', 'images')

    # file name and path
    file_name = df.symbol.unique()[0] + '_' + feature + '_wma' + str(N)
    file_path_table = folder_path_table + file_name + '.csv'
    file_path_plot = folder_path_plot + file_name + '.png'

    # Weighted Moving Average (WMA)
    cumsum = (df[weight] * df[feature]).cumsum()
    cumdiv = df[weight].cumsum()
    wma = (cumsum / cumdiv).rolling(window=N).mean()
    df['wma_' + str(N)] = wma.values

    # comparing summary statistics
    summary_wma = wma.dropna().describe()
    summary_feature = df[feature].describe()
    summary_stat = pd.concat([summary_feature, summary_wma], axis=1)
    summary_stat.columns = [feature, 'WMA_' + str(N)]
    summary_stat.to_csv(file_path_table)

    # plot
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=df.symbol.unique()[0] + ' \n' + feature + ' vs WMA ' + str(N),
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    ax.plot(df['timestamp_CET'], df[feature])
    ax.plot(df['timestamp_CET'], wma)
    fig.text(0.06, 0.5, feature, ha='center', va='center', rotation='vertical')
    ax.margins(x=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend([feature, 'WMA ' + str(N)], bbox_to_anchor=(.5, 0.03),
              loc="lower center", bbox_transform=fig.transFigure, ncol=2, frameon=False)
    plt.savefig(file_path_plot, dpi=160)  # save fig
    plt.close()

    print(feature + ' and WMA ' + str(N) + ' comparison: DONE')
    return df


# 1.h. 15min OHLC
def ohlc(df, time_interval, local_folder):

    # import modules
    import finplot as fplt
    import os

    # folder path
    folder_path = os.path.abspath(os.getcwd())
    folder_path_table = folder_path.replace(local_folder, 'reports/task1/tables/')
    file_path = folder_path_table + 'OHLC_data.csv'

    # derive OHLC data with
    df = df.set_index('timestamp_CET')
    df_ohlc = df['price'].resample(time_interval).ohlc()
    df_ohlc.to_csv(file_path)

    # OHLC plot
    ax = fplt.create_plot(df.symbol.unique()[0], rows=1, maximize=True)
    ax.setLabel('left', 'Price')
    fplt.candlestick_ochl(df_ohlc[['open', 'close', 'high', 'low']])
    fplt.show()
    fplt.close()

    print('Task 1.h: derive my own ' + time_interval + ' OHLC data from your cleaned dataset in 1.d: DONE')
    return df_ohlc


# 1.i. Derive your own 15min median and amount weighted price from your cleaned dataset in 1.d).
#           Compare the median, amount weighted price and the Close price of 1.f.) with each other.
def comparison(df, time_interval, feature, weight, N, local_folder):

    # import modules
    import pandas as pd
    import os
    import matplotlib.pyplot as plt

    # folder path
    folder_path = os.path.abspath(os.getcwd())
    folder_path_table = folder_path.replace(local_folder, 'reports/task1/tables/')
    folder_path_plot = folder_path_table.replace('tables', 'images')

    # file name and path
    file_name = df.symbol.unique()[0] + '_summary'
    file_path_table = folder_path_table + file_name + '.csv'
    file_path_plot = folder_path_plot + file_name + '.png'

    # min interval
    df.reset_index(inplace=True)
    df.set_index('timestamp_CET', inplace=True)
    price_median = df['price'].resample(time_interval).median()
    df['median_' + time_interval] = price_median
    cumsum = (df[weight] * df[feature]).cumsum()
    cumdiv = df[weight].cumsum()
    wma = (cumsum / cumdiv).rolling('15min').mean()
    df['wma_' + time_interval] = wma

    # summary statistics
    summary_price_median = price_median.describe()
    summary_wma_min = df['wma_' + time_interval].describe()
    summary_wma_N = df['wma_' + str(N)].describe()
    summary_close_price = df['price'].describe()
    summary = pd.DataFrame([summary_price_median, summary_wma_min, summary_wma_N, summary_close_price]).T
    summary.columns = [time_interval + '_close_price_median', 'wma_' + time_interval, 'wma_' + str(N), 'close_price']
    summary.to_csv(file_path_table)

    # plot
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=df.symbol.unique()[0] +
               ' \n Close Price, ' + time_interval + 'Median Price, ' + 'WMA ' + time_interval + ', WMA ' + str(N),
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    ax.plot(df['price'])
    ax.plot(df['median_' + time_interval])
    ax.plot(df['wma_' + time_interval])
    ax.plot(df['wma_' + str(N)])
    fig.text(0.06, 0.5, 'Price', ha='center', va='center', rotation='vertical')
    ax.margins(x=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(['Close Price', 'Price Median ' + time_interval, 'WMA ' + time_interval, 'WMA ' + str(N)],
              bbox_to_anchor=(.5, 0.03), loc="lower center", bbox_transform=fig.transFigure, ncol=4, frameon=False)
    plt.savefig(file_path_plot, dpi=160)  # save fig
    plt.close()

    print('Task 1.i: final comparison: DONE')

    return df


# function to solve task 1
def task_1(local_folder, data_folder, feature, weight, N, time_interval):

    # import modules
    import os
    import matplotlib.pyplot as plt

    # 1.a. merge dataset
    df = merge_dataset(local_folder=local_folder, data_folder=data_folder)

    # 1.b. clean data
    df = clean_data(merged_df=df)

    # 1.c. Timestamp CET
    df = timestamp_CET(df=df)

    # 1.d. plot: price and amount
    price_amount_plot(df=df, local_folder=local_folder)

    # 1.e. Clean the outliers in variable amount
    remove_outliers(df=df, feature=weight, local_folder=local_folder)

    # 1.f. 100 point simple moving average
    df = sma(df=df, feature=feature, N=N, local_folder=local_folder)

    # 1.g. 100 point weighted moving average
    df = wma(df=df, feature=feature, weight=weight, N=N, local_folder=local_folder)
    folder_path = os.path.abspath(os.getcwd())
    folder_path_plot = folder_path.replace(local_folder, 'reports/task1/images/')
    file_name = df.symbol.unique()[0] + '_' + feature + 'sma_and_wma' + str(N)
    file_path_plot = folder_path_plot + file_name + '.png'

    # plot
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=df.symbol.unique()[0] + ' \n' + feature + ' vs SMA ' + str(N) + ' vs WMA ' + str(N),
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    ax.plot(df['timestamp_CET'], df[feature])
    ax.plot(df['timestamp_CET'], df['sma_' + str(N)])
    ax.plot(df['timestamp_CET'], df['wma_' + str(N)])
    fig.text(0.06, 0.5, feature, ha='center', va='center', rotation='vertical')
    ax.margins(x=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend([feature, 'SMA ' + str(N), 'WMA ' + str(N)], bbox_to_anchor=(.5, 0.03),
              loc="lower center", bbox_transform=fig.transFigure, ncol=3, frameon=False)
    plt.savefig(file_path_plot, dpi=160)  # save fig
    plt.close()

    print('Task 1.g: ' + feature + ' and SMA  and WMA ' + str(N) + ' comparison: DONE')

    # 1.h. 15min OHLC
    df_ohlc = ohlc(df=df, time_interval=time_interval, local_folder=local_folder)

    # 1.i. final comparison
    df = comparison(df=df, time_interval=time_interval, feature=feature, weight=weight, N=N, local_folder=local_folder)

    # save results
    folder_path = os.path.abspath(os.getcwd())
    file_path = folder_path.replace(local_folder, 'reports/task1/tables/final_data.csv')
    df.to_csv(file_path)

    return df