#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
import pandas as pd
from pandas import DataFrame as DF

import os
from os.path import join
import math
import driving_data as dd
import repo_env
import constants as C


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


def __to_feature(car, feature):
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
            return min(sols) * 1000. / 3600.
        else:
            return max(sols) * 1000. / 3600.

    def ttcp():
        if vx == 0 and vy == 0:
            return float('inf')
        else:
            return -(x * vx + y * vy) * 1000 / (vx ** 2 + vy ** 2) / 3600  # km/h*1000/3600 = m/s

    def dtcp():
        if vx == 0 and vy == 0:
            return float('inf')
        else:
            return abs(x * vy - y * vx) / math.sqrt(vx ** 2 + vy ** 2)

    if feature == 'ttac':
        if dtcp() < C.CAR_CIRCLE_RADIUS * 2:
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
            return calc_ttc(x, vx, ax, C.CAR_WIDTH)
        except ValueError:
            return np.float('inf')
    elif feature == "ttcy":
        try:
            return calc_ttc(y, vy, ay, C.CAR_LENGTH)
        except ValueError:
            return np.float('inf')
    elif feature == "dist":
        return math.sqrt(x ** 2 + y ** 2)
    elif feature.value == "deg":
        return math.atan2(y, x) / math.pi * 180


## TODO driving dataと同じコード。ddのほうは読み込み時間かかって使いにくい。csv読み込みは別にメソッドで用意したほうがいいのかもしれない。
def __get_cars(sur_row):
    sur_row = np.array(sur_row)
    cars = sur_row.reshape(int(sur_row.shape[0] / 6), 6).tolist()
    return filter(lambda car: not all([item == 0 for item in car]), cars)


## TODO driving dataと同じコード。ddのほうは読み込み時間かかって使いにくい。csv読み込みは別にメソッドで用意したほうがいいのかもしれない。
def __to_eachcar_list(sur):
    return [__get_cars(sur_row) for sur_row in sur]


(Front, Center, Rear) = range(3)
(Left, Center, Right) = range(3)
(Relx, Rely, Relvx, Relvy, Accx, Accy) = range(6)


def __add_accel(sur):
    frame_numbers = sur.shape[0]
    car_numbers_mult4 = sur.shape[1]

    sur_each_car_list = [sur[:, i * 4:i * 4 + 4] for i in range(int(car_numbers_mult4 / 4))]

    sur_each_car_list_with_accel = []
    for sur_each_car in sur_each_car_list:
        current = sur_each_car[1:frame_numbers, Relvx:Relvy + 1]
        previous = sur_each_car[0:frame_numbers - 1, Relvx:Relvy + 1]

        # 1フレーム目の加速度は[0, 0]
        accel = np.r_[[[0, 0]], current - previous]

        sur_each_car_list_with_accel.append(np.c_[sur_each_car, accel])

    return np.hstack(sur_each_car_list_with_accel)


def __to_index(car):
    """
    car to index
    :param car:
    :return:
    """
    if car[Rely] > C.CAR_LENGTH / 2:
        y = Front
    elif C.CAR_LENGTH / 2 >= car[Rely] >= - (C.CAR_LENGTH / 2):
        y = Center
    else:
        y = Rear
    if car[Relx] > C.CAR_WIDTH / 2:
        x = Right
    elif C.CAR_WIDTH / 2 >= car[Relx] >= -(C.CAR_WIDTH / 2):
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
            feature_value = __to_feature(car, feature)
            nine_neighbor[y][x] = feature_value
        afeature.append([math.atan(neigh) for neigh in nine_neighbor.reshape(9)])
    return np.array(afeature)


def __to_30frames(feature_df, label):
    # TODO このDataFrameからNumpyにして戻す感じなんとかしたい。DFのままできるのなら
    df_columns = feature_df.columns
    feature_np = feature_df.as_matrix()

    col_num = C.NUM_OF_SEQUENCE * feature_np.shape[1]
    row_num = feature_np.shape[0] - C.NUM_OF_SEQUENCE + 1

    df30 = [feature_np[i:C.NUM_OF_SEQUENCE + i, :][::-1].transpose().reshape(col_num) for i in range(row_num)]
    label = label[C.NUM_OF_SEQUENCE - 1:]

    df30_columns = ['{0}_{1}frames_before'.format(df_column, i) for df_column in df_columns for i in
                    range(C.NUM_OF_SEQUENCE)]

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

    lc_ser = lc_ser[C.PRED_FRAME:]
    df_as_np = df_as_np[:-1 * C.PRED_FRAME]

    return DF(df_as_np, columns=df_columns), lc_ser


if __name__ == '__main__':
    # dataframeとnparrayをもうちょっとわかりやすい基準に分けたい。

    import time

    label_list = []
    feature_list = []
    for i, df in enumerate(dd.behavior_key_nparrays_value.values()):
        start = time.time()
        print(i)
        sur = __add_accel(df['sur'].as_matrix())


        drv_df = df['drv'][:][['brake', 'gas', 'vel', 'steer']]

        # クソコード
        hl = pd.Series(np.float('NaN') if item == 'Null' else float(item)for item in df['roa']['host_lane']).interpolate()
        ln = pd.Series(np.float('NaN') if item == 'Null' else float(item)for item in df['roa']['lane_number']).interpolate()
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
        #feature_df = pd.concat([drv_df, roa_df, sur_feature_df], axis=1, join='inner')

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

    np.savez_compressed(join(repo_env.DATA_DIR,'X'), X=X)

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
    fv_train_0 = f_train[l_train == 0][:sample*5, :]
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
