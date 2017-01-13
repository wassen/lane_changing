#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from os.path import join
from os.path import exists
import numpy as np
import pandas as pd
import sklearn
from pandas import DataFrame as DF
from sklearn.model_selection import train_test_split
import constants as c
import repo_env
from os import listdir as ls

# import driving_data as dd

front_center = 'front_center'
rear_right = 'rear_right'
types = ('drv', 'roa', 'sur')
columns = ('label',
           'gas', 'brake', 'steer',
           'front_center_distance', 'front_center_relvy', 'front_center_ittc_2ndy',
           'rear_right_distance', 'rear_right_relvy', 'rear_right_ittc_2ndy')


class DataEachLC:
    @classmethod
    def load_each_lc(cls, deci_second):
        def load_right_divide_9000():

            def load(behavior, lc_series_num, type):
                return pd.read_pickle(join(repo_env.DATA_DIR, 'divide_9000', behavior, 'right', lc_series_num, type))

            type_df_dict_list = []
            for behavior in sorted(ls(join(repo_env.DATA_DIR, 'divide_9000'))):
                for lc_series_num in sorted(ls(join(repo_env.DATA_DIR, 'divide_9000', behavior, 'right'))):
                    type_df_dict_list.append({tYpe: load(behavior, lc_series_num, tYpe) for tYpe in types})
            return type_df_dict_list

        for j, type_df_dict in enumerate(load_right_divide_9000()):
            start_i = start_index(type_df_dict['roa']['lc'])['right'][0]
            if start_i < deci_second:
                first_of_array = 0
            else:
                first_of_array = start_i - deci_second

            drv_10_sec = type_df_dict['drv'][first_of_array:start_i]
            roa_10_sec = type_df_dict['roa'][first_of_array:start_i]
            sur_10_sec = add_accel(type_df_dict['sur'][first_of_array:start_i])
            length = len(drv_10_sec)
            features = []
            if length != 105:
                continue
            # 車線変更開始前の特定にタイミングにおける車両を追い続ける
            index_of_detect_car = 5
            fixed_f_c_car = specific_nearest_car2(get_cars(sur_10_sec[index_of_detect_car]), "front_center")
            fixed_r_r_car = specific_nearest_car2(get_cars(sur_10_sec[index_of_detect_car]), "rear_right")

            for i, (drv, roa, sur) in enumerate(zip(drv_10_sec.iterrows(), roa_10_sec.iterrows(), sur_10_sec)):

                feature = []
                drv = drv[1]

                cars = get_cars(sur)
                # f_c_car = specific_nearest_car2(cars, front_center)
                # r_r_car = specific_nearest_car2(cars, rear_right)
                f_c_car = cars.get(fixed_f_c_car[0], [])
                r_r_car = cars.get(fixed_r_r_car[0], [])

                def feature_if_exist(car, feature):
                    if len(car) == 0:
                        return None
                    elif feature == 'vy':
                        return car[3]
                    else:
                        return to_feature(car, feature)

                feature.append("{}frame_before".format(length - i))
                # 逆数取る特徴を定義
                feature.append(drv['gas'])
                feature.append(drv['brake'])
                feature.append(drv['steer'])
                feature.append(feature_if_exist(f_c_car, Features.Distance))
                feature.append(feature_if_exist(f_c_car, 'vy'))

                f_c_car_ttcy = feature_if_exist(f_c_car, Features.TimeToCollisionY)
                if f_c_car_ttcy == 0:
                    f_c_car_ttcy = None

                r_r_car_ttcy = feature_if_exist(r_r_car, Features.TimeToCollisionY)
                if r_r_car_ttcy == 0:
                    r_r_car_ttcy = None

                feature.append(1 / f_c_car_ttcy if f_c_car_ttcy is not None else None)
                feature.append(feature_if_exist(r_r_car, Features.Distance))
                feature.append(feature_if_exist(r_r_car, 'vy'))
                feature.append(1 / r_r_car_ttcy if r_r_car_ttcy is not None else None)
                features.append(feature)

                # pd.Series([
                #     length - i,
                #     drv['gas'],
                #     drv['brake'],
                #     drv['steer'],
                #     feature_if_exist(f_c_car, dT.Features.Distance),
                #     feature_if_exist(f_c_car, 'vy'),
                #     1 / f_c_car_ttcy,
                #     feature_if_exist(r_r_car, dT.Features.Distance),
                #     feature_if_exist(r_r_car, 'vy'),
                #     1 / r_r_car_ttcy,
                # ], index=columns)

                # , dropna=True
                # Noneの点が消えるようになってるが、ヒストグラムのところでバグが出る。

            plot_data = pd.DataFrame(features, columns=columns)
            del (features)
            # green_to_red = sns.diverging_palette(145, 10, n=100, center="dark")  # , s=70, l=40, n=3
            # ax = sns.pairplot(pd.DataFrame(features, columns=columns), hue="label", palette=green_to_red[first_of_color_palette:])
            # ax._legend.remove()
            yield plot_data

    def pca_all(self):
        pass

    # def filter_complete(self):
    #     self.data = (
    #         one_lc for one_lc in self.data
    #         if one_lc.dropna().shape[0] == self.deci_second / self.frame_rate
    #     )
    #
    #     # filtered_data_list = filter(
    #     #     lambda plot_data: plot_data.dropna().shape[0] == self.deci_second / self.frame_rate,
    #     #     data_list
    #     # )

    def __add_prev_diff(self, df):
        for i in df:
            n = [None]
            n.extend(df[:-1][i])
            df["prev_{}".format(i)] = n
            kwargs = {"diff_{}".format(i): lambda df: df[i] - df["prev_{}".format(i)]}
            df = df.assign(**kwargs)
        return df.dropna().set_index(np.arange(0, self.deci_second - self.frame_rate, self.frame_rate))

    @staticmethod
    def dfinlist_to_nparray3d(it):
        return np.array([np.array(df) for df in it])

    @staticmethod
    def nparray3d_to_2d(it):
        shape = it.shape
        return it.reshape(shape[0] * shape[1], shape[2])

    def train_test_for_bayes(self, num=5):
        def divided_data(it, num):
            shuffle_it = np.random.permutation([np.array(item) for item in it])
            center = len(shuffle_it) / num
            return shuffle_it[:center], shuffle_it[center:]

        test, train = divided_data(self.extract_data, num)
        return train, test

    def mean_and_cov_train(self):
        train = self.train

    def __init__(self, **kwargs):
        self.deci_second = deci_second = kwargs.get("deci_second", 105)
        self.frame_rate = frame_rate = kwargs.get("frame_rate", 5)
        self.features = features = kwargs.get("features", columns)
        self.diffs = ["diff_{}".format(feature) for feature in features]
        self.prevs = ["prev_{}".format(feature) for feature in features]
        self.extract_data = None

        pickle_path = repo_env.path("data",
                                    "{0[0]}_{0[1]}_{1}second_{2}frame_rate_each_df_list.pickle".
                                    format(features, deci_second / 10., frame_rate)
                                    )

        if exists(pickle_path):
            self.data = pd.read_pickle(pickle_path)
        else:
            # load->5刻みに、特定の特徴を抜き出す->全フレーム揃ってるやつだけ->0.5sec前の値と、それの差分の列を追加
            self.data = self.__class__.load_each_lc(deci_second)
            self.data = (data[::frame_rate][features] for data in self.data)
            self.data = (
                one_lc for one_lc in self.data
                if one_lc.dropna().shape[0] == self.deci_second / self.frame_rate
            )

            # 105フレーム揃ってるやつだけ抽出。意図せず外れてしまっているやつを直したい。
            # print("{}フレーム揃ってる車線変更データ数は{}個です".format(self.frame_rate, len(self.data)))
            self.data = [
                self.__add_prev_diff(data) for data in self.data
                ]

            pd.to_pickle(self.data, pickle_path)

    def extract_columns(self, columns):
        self.extract_data = [
            data[columns]
            for data in self.data
            ]
        return self.extract_data

    def prev_names(self):
        return ["prev_{}".format(feature) for feature in self.features]

    def diff_names(self):
        return ["diff_{}".format(feature) for feature in self.features]

    def divide(self, iterable, train_percentage=0.2):
        pass
        len(iterable)
        return iterable

    def cv_each_trial(self, ):
        pass


def get_columns_combinations():
    from itertools import combinations
    return combinations(columns[1:], 2)


def specific_nearest_car(cars, cartype):
    half_width = c.LANE_WIDTH / 2.
    if cartype == 'front_center':
        def cond(car):
            # ここの条件はもうちょい考えたほうがいいかも？
            return -half_width < car[0] < half_width and car[1] > 0
    elif cartype == 'rear_right':
        def cond(car):
            return half_width < car[0] < 3 * half_width and car[1] < 0

    front_cars = filter(cond, cars)
    front_car = sorted(front_cars, key=lambda x: to_feature(x, Features.Distance))
    if len(front_car) == 0:
        return []
    else:
        return front_car


def specific_nearest_car2(cars, cartype):
    half_width = c.LANE_WIDTH / 2.
    if cartype == 'front_center':
        def cond(car_value):
            # ここの条件はもうちょい考えたほうがいいかも？
            return -half_width < car_value[0] < half_width and car_value[1] > 0
    elif cartype == 'rear_right':
        def cond(car_value):
            return half_width < car_value[0] < 3 * half_width and car_value[1] < 0
    # ここがめんどくさいな
    front_cars = {key: cars[key] for key in cars if cond(cars[key])}
    front_car = sorted(front_cars.items(), key=lambda item: to_feature(item[1], Features.Distance))
    if len(front_car) == 0:
        return [None, []]
    else:
        return front_car[0]


def next_lc(num, start_index):
    def min_with_st(ite, st):
        l = filter(lambda x: x > st, ite)
        if len(l) == 0:
            return float("inf")
        else:
            return min(l)

    r = min_with_st(start_index['right'], num)
    l = min_with_st(start_index['left'], num)

    if r < l:
        return r, 'right'
    elif l < r:
        return l, 'left'
    else:
        return float('inf'), 'None'


def start_index(l):
    def onetotwo(n):
        if n == 0:
            return 0
        elif n == 1:
            return 2
        elif n == -1:
            return 1
        else:
            raise

    l = np.array(l)
    l = map(onetotwo, l)
    l = l - np.append(0, l[:-1])
    if l[0] == 0:
        l[0] = -10
    return {'str': list(np.where(l < 0)[0]),
            'right': list(np.where(l == 2)[0]),
            'left': list(np.where(l == 1)[0]), }


class Features():
    # note:name = value
    TimeToActualCollision = "ttac"
    TimeToClosestPoint = "ttcp"
    DistanceToClosestPoint = "dtcp"
    TimeToCollisionX = "ttcx"
    TimeToCollisionY = "ttcy"
    Distance = "dist"
    Degree = "deg"
    Label = "label"

    # def unit(self):
    #     if 'Time' in self.name:
    #         return 'sec'
    #     elif "Distance" in self.name:
    #         return 'm'
    #     elif self.name == "Degree":
    #         return 'deg'


def to_feature(car, feature):
    x = float(car[0])
    y = float(car[1])
    vx = float(car[2])
    vy = float(car[3])
    ax = float(car[4])
    ay = float(car[5])

    def solve_quad(a, b, c):
        if a == 0:
            return - c / b, - c / b
        else:
            return (-b + math.sqrt(pow(b, 2) - 4 * a * c)) / 2 / a, (-b - math.sqrt(pow(b, 2) - 4 * a * c)) / 2 / a

    def calc_ttc(x, v, a, car_size):
        sols = solve_quad(1. / 2. * a, v, (x - car_size * np.sign(x)))
        if sols[0] > 0 and sols[1] > 0:
            return min(sols)
        else:
            return max(sols)

    def ttcp():
        if vx == 0 and vy == 0:
            return float('inf')
        else:
            return -(x * vx + y * vy) / (vx ** 2 + vy ** 2)  # km/h*1000/3600 = m/s

    def dtcp():
        if vx == 0 and vy == 0:
            return float('inf')
        else:
            return abs(x * vy - y * vx) / math.sqrt(vx ** 2 + vy ** 2)

    if feature == 'ttac':
        if dtcp() < c.CAR_CIRCLE_RADIUS * 2:
            return ttcp()
        else:
            return float('inf')

    elif feature == 'ttcp':
        return ttcp()
    elif feature == "dtcp":
        return dtcp()
    elif feature == "ttcx":
        # TODO NaNが出る問題
        try:
            return calc_ttc(x, vx, ax, c.CAR_WIDTH)
        except ValueError:
            return np.float('inf')
    elif feature == "ttcy":
        try:
            return calc_ttc(y, vy, ay, c.CAR_LENGTH)
        except ValueError:
            return np.float('inf')
    elif feature == "dist":
        return math.sqrt(x ** 2 + y ** 2)
    elif feature.value == "deg":
        return math.atan2(y, x) / math.pi * 180


## TODO driving dataと同じコード。ddのほうは読み込み時間かかって使いにくい。csv読み込みは別にメソッドで用意したほうがいいのかもしれない。
def get_cars(sur_row):
    sur_row = np.array(sur_row)
    cars = sur_row.reshape(int(sur_row.shape[0] / 6), 6).tolist()
    return {i: car for i, car in enumerate(cars) if not all([item == 0 for item in car])}


## TODO driving dataと同じコード。ddのほうは読み込み時間かかって使いにくい。csv読み込みは別にメソッドで用意したほうがいいのかもしれない。
def __to_eachcar_list(sur):
    return [get_cars(sur_row) for sur_row in sur]


(Front, Center, Rear) = range(3)
(Left, Center, Right) = range(3)
(Relx, Rely, Relvx, Relvy, Accx, Accy) = range(6)


def add_accel(sur):
    sur = np.array(sur)
    frame_numbers = sur.shape[0]
    car_numbers_mult4 = sur.shape[1]

    sur_each_car_list = [sur[:, i * 4:i * 4 + 4] for i in range(int(car_numbers_mult4 / 4))]

    sur_each_car_list_with_accel = []
    for sur_each_car in sur_each_car_list:
        current = sur_each_car[1:frame_numbers, Relvx:Relvy + 1]
        previous = sur_each_car[0:frame_numbers - 1, Relvx:Relvy + 1]

        # 1フレーム目の加速度は[0, 0]
        accel = np.r_[[[0, 0]], 10 * (current - previous)]

        sur_each_car_list_with_accel.append(np.c_[sur_each_car, accel])

    return np.hstack(sur_each_car_list_with_accel)


def __to_index(car):
    """
    car to index
    :param car:
    :return:
    """
    if car[Rely] > c.CAR_LENGTH / 2:
        y = Front
    elif c.CAR_LENGTH / 2 >= car[Rely] >= - (c.CAR_LENGTH / 2):
        y = Center
    else:
        y = Rear
    if car[Relx] > c.CAR_WIDTH / 2:
        x = Right
    elif c.CAR_WIDTH / 2 >= car[Relx] >= -(c.CAR_WIDTH / 2):
        x = Center
    else:
        x = Left

    return y, x


def __sur_feature(sur, feature):
    # naming
    afeature = []
    for each_car in __to_eachcar_list(sur):
        nine_neighbor = np.ones(9).reshape(3, 3) * np.float('inf')
        for car in each_car:
            y, x = __to_index(car)
            feature_value = to_feature(car, feature)
            nine_neighbor[y][x] = feature_value
        afeature.append([math.atan(neigh) for neigh in nine_neighbor.reshape(9)])
    return np.array(afeature)


def __to_30frames(feature_df, label):
    # TODO このDataFrameからNumpyにして戻す感じなんとかしたい。DFのままできるのなら
    df_columns = feature_df.columns
    feature_np = feature_df.as_matrix()

    col_num = c.NUM_OF_SEQUENCE * feature_np.shape[1]
    row_num = feature_np.shape[0] - c.NUM_OF_SEQUENCE + 1

    df30 = [feature_np[i:c.NUM_OF_SEQUENCE + i, :][::-1].transpose().reshape(col_num) for i in range(row_num)]
    label = label[c.NUM_OF_SEQUENCE - 1:]

    df30_columns = ['{0}_{1}frames_before'.format(df_column, i) for df_column in df_columns for i in
                    range(c.NUM_OF_SEQUENCE)]

    return DF(df30, columns=df30_columns), label


def __delete_lc_after3(feature_df, lc_list):
    rlc_state = 0
    llc_state = 0
    del_index = []
    for i, lc in enumerate(lc_list):

        if lc == 0:
            rlc_state = 0
            llc_state = 0
        elif lc == 1:
            rlc_state += 1
            llc_state = 0
        elif lc == -1:
            llc_state += 1
            rlc_state = 0
        else:
            print('lcに0,1,-1の文字が含まれています。')
        if rlc_state > 3:
            del_index.append(i)
        elif llc_state > 3:
            del_index.append(i)
    df_columns = feature_df.columns
    df_as_np = feature_df.as_matrix()
    deleted_np = np.delete(df_as_np, del_index, 0)
    deleted_lc = np.delete(list(lc_list), del_index, 0)
    return DF(deleted_np, columns=df_columns), deleted_lc


def __shift_pred_frames(feature_df, lc_ser):
    df_columns = feature_df.columns
    df_as_np = feature_df

    lc_ser = lc_ser[c.PRED_FRAME:]
    df_as_np = df_as_np[:-1 * c.PRED_FRAME]

    return DF(df_as_np, columns=df_columns), lc_ser


if __name__ == '__main__':
    # dataframeとnparrayをもうちょっとわかりやすい基準に分けたい。

    import time

    label_list = []
    feature_list = []
    for i, df in enumerate(dd.behavior_and_drivingdata.values()):
        start = time.time()
        print(i)
        sur = add_accel(df['sur'].as_matrix())

        drv_df = df['drv'][:][['brake', 'gas', 'vel', 'steer']]

        # クソコード
        hl = pd.Series(
            np.float('NaN') if item == 'Null' else float(item) for item in df['roa']['host_lane']).interpolate()
        ln = pd.Series(
            np.float('NaN') if item == 'Null' else float(item) for item in df['roa']['lane_number']).interpolate()
        roa_df = pd.DataFrame([hl, ln]).transpose()

        feature_name = Features.TimeToCollisionX
        columns = ['{0}_{1}_{2}'.format(feature_name, y, x) for y in ['Front', 'Center', 'Rear'] for x in
                   ['Left', 'Center', 'Right']]
        print('TTCまえ')
        print(time.time() - start)
        sur_feature = __sur_feature(sur, Features.TimeToActualCollision)
        print('TTC後')
        print(time.time() - start)

        sur_feature_df = DF(sur_feature, columns=columns)


        # feature_df = pd.concat([drv_df, roa_df, sur_feature_df], axis=1, join='inner')

        # kusoko-dohajimari
        def __relv(sur):
            # naming
            afeature = []
            for each_car in __to_eachcar_list(sur):
                if len(each_car) == 0:
                    t = 0
                else:
                    t = np.average([car[3] for car in each_car])
                afeature.append(t)
            return np.array(afeature)


        sur_feature2 = __relv(sur)
        sur_feature2_df = DF(sur_feature2, columns=['new!'])
        feature_df = pd.concat([drv_df, roa_df, sur_feature_df, sur_feature2_df], axis=1, join='inner')

        # kusoko-doowari

        # ここfeature_dfのメソッド拡張してやれたら見た目すっきりしそう。なんにせよ再代入が嫌
        feature_df, lc_ser = __to_30frames(feature_df, df['roa']['lc'])
        feature_df, lc_ser = __delete_lc_after3(feature_df, lc_ser)
        feature_df, lc_ser = __shift_pred_frames(feature_df, lc_ser)

        # 暫定 TTAC以外でNanがあるらしい・・・勘弁してくれ。
        feature_list.append(feature_df.fillna(0).as_matrix())
        label_list.append(lc_ser)
        print('おしまい')
        print(time.time() - start)
        # 最後にデータを確認
        # TODO データの保存。
        # TODO ProgressBarを2に対応させる。
        # label_list.append(df['roa']['lc'])
        # TODO vstack all behavior
        # TODO lane_number, self_lane、左に車線があるかなどをを特徴に追加lane_numberが逆なことに注意
        # 加速度って10倍しなきゃ？
        # Keepの内容見よう。

    X = np.vstack(feature_list)
    Y = np.hstack(label_list)

    # import pandas as pd
    # addaccel(pd.DataFrame([[1, 2, 3, 4, 1, 1, 3, 4, 5, 3, 2, 1],
    #                       [1, 3, 3, 5, 1, 1, 2, 2, 4, 5, 1, 1],
    #                       [1, 3, 4, 2, 1, 1, 4, 4, 4, 5, 1, 1]]).as_matrix())
    print('ほんとにおしまい')

    np.savez_compressed(join(repo_env.DATA_DIR, 'X'), X=X)

    l = f_train, f_test, l_train, l_test = train_test_split(X, Y, test_size=0.25, train_size=0.75)

    # 120以上がsur_feature
    print(X.shape)
    #
    print(all(np.where(np.isnan(X))[1] > 120))

    print(any(np.isnan(np.array(f_train).reshape(f_train.size))))
    print(any(np.isnan(np.array(f_test).reshape(f_test.size))))
    print(any(np.isnan(np.array(l_train).reshape(l_train.size))))
    print(any(np.isnan(np.array(l_test).reshape(l_test.size))))
    print([np.float('inf') in a for a in l])

    # 1のクラス数に合わせる、拡張して、全てのクラスのminを計算したい
    fv_train_1 = f_train[l_train == 1]
    fv_train_m1 = f_train[l_train == -1]
    sample = min(fv_train_1.shape[0], fv_train_m1.shape[0])

    fv_train_1 = f_train[l_train == 1][:sample, :]
    fv_train_0 = f_train[l_train == 0][:sample * 5, :]
    fv_train_m1 = f_train[l_train == -1][:sample, :]
    f_train = np.r_[fv_train_m1, fv_train_0, fv_train_1]
    l_train = np.r_[np.ones(sample) * -1, np.zeros(sample * 5), np.ones(sample)]


    def normalize(train, test):
        scaler = sklearn.preprocessing.MinMaxScaler()
        return scaler.fit_transform(train), scaler.transform(test)


    # これもメソッド拡張したい。
    f_train, f_test = normalize(f_train, f_test)

    np.savez_compressed(join(repo_env.DATA_DIR, 'train_test_feature_label')
                        , f_train=f_train, l_train=l_train, f_test=f_test, l_test=l_test)
