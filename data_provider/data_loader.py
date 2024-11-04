import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask
     
class VitalDBLoader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100, seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path

        self.flag = flag
        
        # Initialize scalers for each feature to be standardized
        self.scaler_bts = StandardScaler()
        self.scaler_hrs = StandardScaler()
        self.scaler_dbp = StandardScaler()
        self.scaler_mbp = StandardScaler()
        self.scaler_prediction_mbp = StandardScaler()

        self.__read_data__()
    
    def __read_data__(self):
        if self.flag == 'train':
            df_raw = pd.read_csv(os.path.join(self.root_path, 'vitaldb_train_data.csv'))
            df_raw = df_raw[:int(len(df_raw)*(self.percent/100))]
        elif self.flag == 'val':
            df_raw = pd.read_csv(os.path.join(self.root_path, 'vitaldb_val_data.csv'))
            df_raw = df_raw[:int(len(df_raw) *(self.percent/100))]
        elif self.flag == 'test':
            df_raw = pd.read_csv(os.path.join(self.root_path, 'vitaldb_test_data.csv'))

        # 对csv数据进行预处理
        self.__preprocess_csv__(df_raw)

    def __preprocess_csv__(self, data):

        # 数据预处理前总数据
        print("源数据长度：", len(data))
        label_counts = data['label'].value_counts(normalize=True) * 100
        print("处理前的Label分布 (%):")
        print(label_counts)

        # 定义处理序列数据的函数，直接通过空格拆分并转换为浮点数列表，且完成重采样
        def parse_sequence(sequence_str, skip_rate=0, sample_type='avg_sample'):
            try:
                sequence_list = sequence_str.split()
                sequence_array = np.array([np.nan if x == 'nan' else float(x) for x in sequence_list])
                mean_value = round(np.nanmean(sequence_array), 2)
                sequence_array_filled = np.where(np.isnan(sequence_array), mean_value, sequence_array)
                if np.any(np.isnan(sequence_array_filled)):
                    return [] 
                
                def sliding_window_average(time_series, slide_len):
                    if slide_len <= 0:
                        raise ValueError("slide_len must be greater than 0")
                    
                    # 存储滑动窗口的平均值
                    window_averages = []
                    
                    # 遍历序列，按滑动窗口大小取值
                    for i in range(0, len(time_series), slide_len):
                        # 获取当前窗口的值
                        window = time_series[i:i + slide_len]
                        # 计算窗口的平均值并存储
                        window_avg = round(np.nanmean(window), 2)
                        window_averages.append(window_avg)
                    
                    return window_averages

                if skip_rate > 0: # 如果需要重采样
                    if sample_type == 'skip_sample':
                        sequence_array_filled = sequence_array_filled[::skip_rate]
                    elif sample_type == 'avg_sample': #默认按平均值进行采样
                        sequence_array_filled = sliding_window_average(sequence_array_filled, skip_rate)

                return sequence_array_filled
            except ValueError:
                return [] 
            
        # 初始化 defaultdict
        self.scaler = StandardScaler()
        examples = defaultdict(list)

        for index, row in data.iterrows():
            # if index > 100:
            #     break
            bts = parse_sequence(row['bts'][1:-1], skip_rate=0, sample_type='skip_sample') #采样周期是：2*skip_rate
            hrs = parse_sequence(row['hrs'][1:-1], skip_rate=0, sample_type='skip_sample')
            dbp = parse_sequence(row['dbp'][1:-1], skip_rate=0, sample_type='skip_sample')
            mbp = parse_sequence(row['mbp'][1:-1], skip_rate=0, sample_type='skip_sample')
            prediction_mbp = parse_sequence(row['prediction_mbp'][1:-1], skip_rate=0, sample_type='skip_sample')
            # print(len(bts), len(hrs), len(dbp), len(mbp), len(prediction_mbp))
            if len(bts) != 450 or len(hrs) != 450 or len(dbp) != 450 or\
                len(mbp) != 450 or len(prediction_mbp) != 150:
                continue
            
            examples['caseid'].append(row['caseid'])
            examples['stime'].append(row['stime'])
            examples['ioh_stime'].append(row['ioh_stime'])
            examples['ioh_dtime'].append(row['ioh_dtime'])
            examples['age'].append(row['age']) # np.full(len(bts), row['age'])
            examples['sex'].append(row['sex'])
            examples['bmi'].append(row['bmi'])
            examples['label'].append(row['label'])
            examples['bts'].append(bts)
            examples['hrs'].append(hrs)
            examples['dbp'].append(dbp)
            examples['mbp'].append(mbp)
            examples['prediction_mbp'].append(prediction_mbp)

        # 修正统计处理后的样本数量
        print("处理后的测试样本数量:", len(examples['caseid']))

        # 统计处理后 examples 中 label 列的分布
        label_counts = pd.Series(examples['label']).value_counts(normalize=True) * 100
        print("处理后的Label分布 (%):")
        print(label_counts)

        # # 仅在训练集上进行标准化处理
        # if self.flag == 'train' and self.scale:
        #     print("Fitting scalers on training data...")
        #     self.scaler_bts.fit(examples['bts'])
        #     self.scaler_hrs.fit(examples['hrs'])
        #     self.scaler_dbp.fit(examples['dbp'])
        #     self.scaler_mbp.fit(examples['mbp'])
        #     self.scaler_prediction_mbp.fit(examples['prediction_mbp'])

        # # 对验证集和测试集使用训练集拟合好的scaler进行标准化
        # if self.scale:
        #     print("Transforming data with fitted scalers...")
        #     examples['bts'] = self.scaler_bts.transform(examples['bts'])
        #     examples['hrs'] = self.scaler_hrs.transform(examples['hrs'])
        #     examples['dbp'] = self.scaler_dbp.transform(examples['dbp'])
        #     examples['mbp'] = self.scaler_mbp.transform(examples['mbp'])
        #     examples['prediction_mbp'] = self.scaler_prediction_mbp.transform(examples['prediction_mbp'])

        self.data = examples

    def __getitem__(self, index):
        if self.features == 'S': # 单变量时序预测
            mbp = self.data['mbp'][index]
            seq_x = np.stack([mbp], axis=1)

        else: # 'MS' 'M' 多变量时序预测
            # 提取数据中第 index 行的特征
            bts = self.data['bts'][index]
            hrs = self.data['hrs'][index]
            dbp = self.data['dbp'][index]
            mbp = self.data['mbp'][index]
            # # 直接将列表转换为 NumPy 数组，并进行 stack 操作
            # seq_x = np.stack([bts, hrs, dbp, mbp], axis=1)

            # # 将标量特征扩展到与时间序列相同的长度（seq_len）
            sex = np.full(len(bts), self.data['sex'][index])
            age = np.full(len(bts), self.data['age'][index])
            bmi = np.full(len(bts), self.data['bmi'][index]) 
            seq_x = np.stack([sex, age, bmi, bts, hrs, dbp, mbp], axis=1) 

        # 预测的目标数据是 prediction_mbp 和当前的 mbp，构建 seq_y
        prediction_mbp = self.data['prediction_mbp'][index]
        seq_y = np.concatenate([mbp, prediction_mbp])[:, np.newaxis]

        # 随机生成 seq_x_mark 和 seq_y_mark
        seq_x_mark = np.random.rand(*seq_x.shape)
        seq_y_mark = np.random.rand(*seq_y.shape)
        # seq_x_mark = seq_x
        # seq_y_mark = seq_y

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return len(self.data['caseid'])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

