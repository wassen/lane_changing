#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import unittest

from enum import *
from progre import Progre as pb
import matplotlib
import numpy as np
import os
import sys
import math
import pandas as pd
from sympy import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DataInput(Enum):
    loadVectorData = 'vector'
    loadOriginalData = 'origin'
    readFromCSVData = 'read'


# クラス内クラスとかできるのか。バウンドインナークラス？
# リストじゃなくて辞書で。carとか
# くそみたいな変数名を修正
# pandasどうしよう。surだけ取り出すってのがめんどくさくなりそう。
#



class Dist(IntEnum):
    exFarFront = 0
    farFront = 1
    midFront = 2
    nearFront = 3
    onSelf = 4
    nearRear = 5
    midRear = 6
    farRear = 7
    exFarRear = 8


class Side(IntEnum):
    left = 0
    straight = 1
    right = 2


class Sub(IntEnum):
    x0 = 0
    y0 = 1
    vx = 2
    vy = 3


class Label(IntEnum):
    left_lanechanging = -2
    begin_left_lanechange = -1
    go_straight = 0
    begin_right_lanechange = 1
    right_lanechanging = 2
    braking_and_go_straight = 3


def dividelabelwithbrake(label, brake, threshold=5):
    '''
    直進ラベル(0)をブレーキ踏力のしきい値から0と3に分ける
    '''
    return [Label.braking_and_go_straight.value if b >= threshold and l == Label.go_straight else l for l,b in zip(label, brake)]


class NeighborBlock:
    # 距離と速度だけは無限大のむきを考慮しなきゃいけないから、引数でなんとか処理を分岐

    def __init__(self, *initArray):
        if len(initArray) != 0:
            initArray = np.array(initArray)
            if initArray.size != 27:
                print('argError')
                raise
            else:
                self.featureList = np.array(initArray).reshape(9, 3)
        else:
            self.featureList = np.array(
                [[float('inf') for _ in range(3)] for _ in range(9)])
        self.distList = np.array(
            [[float('inf') for _ in range(3)] for _ in range(9)])

    def add(self, feature, x, y):

        if 30 < y <= 62:
            dist = Dist.exFarFront
        elif 14 < y <= 30:
            dist = Dist.farFront
        elif 6 < y <= 14:
            dist = Dist.midFront
        elif 2 < y <= 6:
            dist = Dist.nearFront
        elif -2 < y <= 2:
            dist = Dist.onSelf
        elif -6 < y <= -2:
            dist = Dist.nearRear
        elif -14 < y <= -6:
            dist = Dist.midRear
        elif -30 < y <= -14:
            dist = Dist.farRear
        elif -64 < y <= -30:
            dist = Dist.exFarRear
        else:
            return

        if 1.5 < x <= 4.5:
            side = Side.right
        elif -1.5 < x <= 1.5:
            side = Side.straight
        elif -4.5 < x <= -1.5:
            side = Side.left
        else:
            return

        if np.linalg.norm([x, y]) < self.distList[dist][side]:
            self.distList[dist][side] = np.linalg.norm([x, y])
            self.featureList[dist][side] = feature

    def get_original_list():
        return [feature for feature in self.featureList.reshape(27)]

    def get_list_atan(self):
        return [math.atan(feature) for feature in self.featureList.reshape(27)]

# メソッドを呼ぶ順番を考えなきゃいけないクラスってどうなん？

def percentile(list1, list2, radius):
    dist = [np.linalg.norm([l1, l2]) for l1, l2 in zip(list1, list2)]
    return sum(np.array(dist) < radius)/len(dist)*100

def start_index(label):

    label = np.array(label)
    labelLists = []
    rLcLabel = list(np.where(label == 1)[0])
    lLcLabel = list(np.where(label == -1)[0])

    rDelList = []
    previous = -2
    for i, l in enumerate(rLcLabel):
        if l == previous + 1:
            rDelList.append(i)
        previous = l
    lDelList = []
    previous = -2
    for i, l in enumerate(lLcLabel):
        if l == previous + 1:
            lDelList.append(i)
        previous = l
    return np.delete(rLcLabel, rDelList), np.delete(lLcLabel, lDelList)

class Features(Enum):
    # note:name = value
    TimeToClosestPoint = "ttcp"
    DistanceToClosestPoint = "dtcp"
    TimeToCollisionX = "ttcx"
    TimeToCollisionY = "ttcy"
    Distance = "dist"
    Degree = "deg"
    Label = "label"

    def unit(self):
        if 'Time' in self.name:
            return 'sec'
        elif "Distance" in self.name:
            return 'm'
        elif self.name == "Degree":
            return 'deg'

class Container:
    REPOSITORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    DATA_PATH_6000 = os.path.join(REPOSITORY_DIR, 'data/Original/6000/')
    DATA_PATH_9000 = os.path.join(REPOSITORY_DIR, 'data/Original/9000/')

    WIDTH_OF_CARS = 2
    LENGTH_OF_CARS = 4

    @classmethod
    def two_features_of_specific_label(cls, feature1_list, feature2_list, label_list, label):
        '''
        特定のラベルを持つ特徴を抜き出します。
        :param feature1_list:
        :param feature2_list:
        :param label_list:
        :param label:
        :return:
        '''
        return (np.array(feature1_list)[np.where(label_list == label)],
                np.array(feature2_list)[np.where(label_list == label)])

    def extract_nearest_car(self, feature_2dim):
        feature_of_nearest_car = []
        dist_2dim = self.feature_with_frames(Features.Distance, load=True)
        for feature_atmoment, dist_atmoment in zip(feature_2dim, dist_2dim):
            if len(feature_atmoment) == 0:
                continue
            min_index = np.argmin(np.array(dist_atmoment))
            feature_of_nearest_car.append(feature_atmoment[min_index])
        return feature_of_nearest_car

    def makeprojectdir(self, dir):
        os.makedirs(os.path.join(self.__class__.REPOSITORY_DIR, dir), exist_ok=True)

    def get_before_and_after_LC_label(label, before=100, after=100):

        label = np.array(label)
        labelLists = []
        lcLabel = list(np.where(label == 1)[0])
        lcLabel.extend(np.where(label == -1)[0])
        if len(lcLabel) == 0:
            return []
        pre = lcLabel[0] - 1
        labelCount = 0

        firstLabelFlag = True
        for i, lc in enumerate(lcLabel):
            if pre != lc - 1:
                tmplc = lcLabel[i - 1]
                for afterlc in np.arange(tmplc + 1, tmplc + 1 + after):
                    if afterlc >= len(label):
                        break
                    else:
                        labelLists[labelCount].append(afterlc)
                labelCount += 1
                firstLabelFlag = True

            if firstLabelFlag == True:
                firstLabelFlag = False
                labelLists.append([])
                for beforelc in np.arange(lc - before, lc):
                    if beforelc < 0:
                        break
                    else:
                        labelLists[labelCount].append(beforelc)
            labelLists[labelCount].append(lc)
            pre = lc

        for afterlc in np.arange(pre + 1, pre + 1 + after):
            if afterlc >= len(label):
                break
            else:
                labelLists[labelCount].append(afterlc)
        # クソcode

        return labelLists

    def show_hist(self, nameOfFeature):

        # コードがひどい

        i = self.featureNames.index(nameOfFeature)
        plt.legend()
        # plt.title("graph Title")
        # plt.xlim([tmp_label[0]/10,tmp_label[len(tmp_label) - 1]]/10)

        le = []
        for j, (label, someFeature) in enumerate(zip(pb.single_generator(self.oneDimVectors), self.twoDimVectors)):
            feature = someFeature[:, i]
            tmp = self.__class__.start_index(label)[0]
            if len(tmp) != 0:
                le.extend(feature[tmp])

        le = np.delete(np.array(le), np.where(np.array(le) == float('inf')))
        le = np.delete(np.array(le), np.where(np.array(le) == float('-inf')))
        plt.hist(le)

        self.makeprojectdir("graph")
        plt.xlabel("Value")
        plt.ylabel(nameOfFeature)
        plt.savefig(os.path.join(self.__class__.REPOSITORY_DIR, 'graph',nameOfFeature + "_r" + '.pdf'))
        plt.clf()

        plt.legend()
        # plt.title("graph Title")
        # plt.xlim([tmp_label[0]/10,tmp_label[len(tmp_label) - 1]]/10)

        le = []
        for j, (label, someFeature) in enumerate(zip(pb.single_generator(self.oneDimVectors), self.twoDimVectors)):
            feature = someFeature[:, i]
            tmp = self.__class__.start_index(label)[1]
            if len(tmp) != 0:
                le.extend(feature[tmp])

        le = np.delete(np.array(le), np.where(np.array(le) == float('inf')))
        le = np.delete(np.array(le), np.where(np.array(le) == float('-inf')))

        plt.hist(le)

        # ファイル名周り未検証
        self.makeprojectdir("graph")

        plt.xlabel("Value")
        plt.ylabel(nameOfFeature)
        plt.savefig(os.path.join(self.__class__.REPOSITORY_DIR,
                                 "graph",
                                 nameOfFeature.replace('\\', "{BSlash}") + "_l" + '.pdf')
                    )
        plt.clf()

    def concat_all_behavior(self, sequence):
        result_list = []
        for name in self.behaviornames:
            result_list.extend(sequence[name])
        return result_list

    def feature_with_frames(self, feature, load=False):
        def save_feature():
            def savetotmp(item, name):
                self.makeprojectdir('tmp')
                np.savez_compressed(os.path.join(self.__class__.REPOSITORY_DIR,
                                                 "tmp",
                                                 "{0}.npz".format(name)
                                                 ),
                                    item=item,
                                    )

            print("tmpフォルダに{}を保存しています。".format(feature.value))
            savetotmp(feature_dict, feature.value)
            print("保存完了しました。")

        def load_feature():
            def loadz(name):
                return np.load(os.path.join(self.__class__.REPOSITORY_DIR, "tmp", "{0}.npz".format(name)))[
                    "item"].tolist()

            return loadz(feature.value)

        if load:
            print("ロード中です。")
            feature_dict = load_feature()
            print("ロード完了しました。")
        else:
            feature_dict = self.feature_subjectdict(feature)
            # self.label_subjectdict_brakefiltered()
            save_feature()
        print('特徴の辞書サイズは' + str(len(feature_dict)))
        featurelist_2dim = self.concat_all_behavior(feature_dict)

        if feature.value == "label":
            b = start_index(featurelist_2dim)
            c = list(b[0])
            c.extend(b[1])
            featurelist_2dim = [a if not (a == 1 or a == -1) or i in c else a * 2 for i, a in enumerate(featurelist_2dim)]

        return featurelist_2dim

    def count_feature_in_circle(self, feature1, feature2, load=False):

        xlist_2dim = self.feature_with_frames(feature1, True)
        ylist_2dim = self.feature_with_frames(feature2, True)
        start_labels = self.feature_with_frames(Features.Label, load)

        xlist = []
        ylist = []
        llist = []

        # 全車両ver
        # for xlist_atmoment, ylist_atmoment, start_label in zip(xlist_2dim, ylist_2dim, start_label):
        #     length_atmoment = len(xlist_atmoment)
        #     if length_atmoment != len(ylist_atmoment):
        #         print("ひとつ目の特徴とふたつ目の特徴において、特定フレームにおけるサイズの差異が検知されました。なんかおかしいです。")
        #     xlist.extend(xlist_atmoment)
        #     ylist.extend(ylist_atmoment)
        #     llist.extend(list(np.ones(length_atmoment)*start_label))

        # 最近傍ver
        # code clone そもそもDistで車を区切るのなんてここでやることじゃない。
        def loadz(name):
            return np.load(os.path.join(self.__class__.REPOSITORY_DIR, "tmp", "{0}.npz".format(name)))["item"].tolist()

        dist_2dim = self.concat_all_behavior(loadz(Features.Distance.value))
        for xlist_atmoment, ylist_atmoment, dist_atmoment, start_label in zip(xlist_2dim, ylist_2dim, dist_2dim,
                                                                              start_labels):
            if len(xlist_atmoment) == 0:
                continue
            min_index = np.argmin(np.array(dist_atmoment))
            xlist.append(xlist_atmoment[min_index])
            ylist.append(ylist_atmoment[min_index])
            llist.append(start_label)

        llist = np.array(llist)
        left = (np.array(xlist)[np.where(llist == Label.begin_left_lanechange)],
                np.array(ylist)[np.where(llist == Label.begin_left_lanechange)])
        straight = (np.array(xlist)[np.where(llist == Label.go_straight)],
                    np.array(ylist)[np.where(llist == Label.go_straight)])
        brake = (np.array(xlist)[np.where(llist == Label.braking_and_go_straight)],
                 np.array(ylist)[np.where(llist == Label.braking_and_go_straight)])
        right = (np.array(xlist)[np.where(llist == Label.begin_right_lanechange)],
                 np.array(ylist)[np.where(llist == Label.begin_right_lanechange)])

        l = np.array([np.linalg.norm((onecar[0], onecar[1])) for onecar in np.array(left).T])
        r = np.array([np.linalg.norm((onecar[0], onecar[1])) for onecar in np.array(right).T])
        s = np.array([np.linalg.norm((onecar[0], onecar[1])) for onecar in np.array(straight).T])
        b = np.array([np.linalg.norm((onecar[0], onecar[1])) for onecar in np.array(brake).T])

        llist = []
        rlist = []
        slist = []
        blist = []

        for ra in np.arange(0,15,0.1):
            # print(len(l))
            # print(len(r))
            # print(len(s))
            # print(len(b))
            # print(len(np.where(l < ra)[0]))
            # print(len(np.where(r < ra)[0]))
            # print(len(np.where(s < ra)[0]))
            # print(len(np.where(b < ra)[0]))
            ra
            ra
            ra
            ra
            llist.append(len(np.where(l < ra)[0])/len(l))
            rlist.append(len(np.where(r < ra)[0])/len(r))
            slist.append(len(np.where(s < ra)[0])/len(s))
            blist.append(len(np.where(b < ra)[0])/len(b))

        alpha = 0.5
        edgecolor = 'none'

        plt.scatter(np.arange(0,15,0.1),llist, color='#FF5A56', alpha=alpha,
                    edgecolor=edgecolor, label="l", linestyle = 'solid', linewidth = 1.0, marker = 'o')
        plt.scatter(np.arange(0,15,0.1),rlist, color='#CCB947', alpha=alpha,
                    edgecolor=edgecolor, label="r", linestyle = 'solid', linewidth = 1.0, marker = 'o')
        plt.scatter(np.arange(0,15,0.1),slist, color='#5E6AFF', alpha=alpha,
                    edgecolor=edgecolor, label="s", linestyle = 'solid', linewidth = 1.0, marker = 'o')
        plt.scatter(np.arange(0,15,0.1),blist, color='#3F993E', alpha=alpha,
                    edgecolor=edgecolor, label="b", linestyle = 'solid', linewidth = 1.0, marker = 'o')

        plt.legend(scatterpoints=int(1 / alpha))  # 薄い時にレジェンドが見えるように数を増やす試み 二乗してもいいか？

        # plt.xlabel("{0}[{1}]".format(feature1.name, feature1.unit()))
        # plt.ylabel("{0}[{1}]".format(feature2.name, feature2.unit()))
        self.makeprojectdir("graph")
        plt.savefig(os.path.join(self.__class__.REPOSITORY_DIR,
                                 "graph",
                                 "graph_of_{0}_and_{1}.png".format(feature1.value, feature2.value)
                                 )
                    )
        plt.clf()

    def show_plot(self, feature1, feature2, xlim = (-12, 12), ylim=(-12, 12), load=False):
        """
        特徴量を二次元でグラフにプロットする。
        :param feature1: 
        :param feature2: 
        :param xlim: 
        :param ylim: 
        :param load: 
        :return: 
        """
        xlist_2dim = self.feature_with_frames(feature1, load)
        ylist_2dim = self.feature_with_frames(feature2, load)
        start_labels = self.feature_with_frames(Features.Label, load)

        xlist = []
        ylist = []
        llist = []

        # 全車両ver
        # for xlist_atmoment, ylist_atmoment, start_label in zip(xlist_2dim, ylist_2dim, start_label):
        #     length_atmoment = len(xlist_atmoment)
        #     if length_atmoment != len(ylist_atmoment):
        #         print("ひとつ目の特徴とふたつ目の特徴において、特定フレームにおけるサイズの差異が検知されました。なんかおかしいです。")
        #     xlist.extend(xlist_atmoment)
        #     ylist.extend(ylist_atmoment)
        #     llist.extend(list(np.ones(length_atmoment)*start_label))

        # 最近傍ver
        # code clone そもそもDistで車を区切るのなんてここでやることじゃない。
        def loadz(name):
            return np.load(os.path.join(self.__class__.REPOSITORY_DIR, "tmp", "{0}.npz".format(name)))["item"].tolist()
        dist_2dim = self.concat_all_behavior(loadz(Features.Distance.value))
        for xlist_atmoment, ylist_atmoment, dist_atmoment, start_label in zip(xlist_2dim, ylist_2dim, dist_2dim, start_labels):
            if len(xlist_atmoment) == 0:
                continue
            min_index = np.argmin(np.array(dist_atmoment))
            xlist.append(xlist_atmoment[min_index])
            ylist.append(ylist_atmoment[min_index])
            llist.append(start_label)
        print(len(llist))
        print(len(xlist))

        llist = np.array(llist)
        left = (np.array(xlist)[np.where(llist == Label.begin_left_lanechange)],
                np.array(ylist)[np.where(llist == Label.begin_left_lanechange)])
        straight = (np.array(xlist)[np.where(llist == Label.go_straight)],
                    np.array(ylist)[np.where(llist == Label.go_straight)])
        brake = (np.array(xlist)[np.where(llist == Label.braking_and_go_straight)],
                    np.array(ylist)[np.where(llist == Label.braking_and_go_straight)])
        right = (np.array(xlist)[np.where(llist == Label.begin_right_lanechange)],
                 np.array(ylist)[np.where(llist == Label.begin_right_lanechange)])
        alpha = 0.50
        edgecolor = 'none'


        # 暫定
        # plt.clf()
        # tmp1 = []
        # tmp2 = []
        # tmp1.extend(right[0])
        # tmp1.extend(left[0])
        # tmp2.extend(right[1])
        # tmp2.extend(left[1])
        # radiuses = np.arange(0, 12, 0.1)
        # print(tmp1)
        # print(tmp2)
        # plt.scatter(radiuses, [percentile(tmp1, tmp2, radius) for radius in radiuses],
        #            color='#2FCDB4',alpha=alpha, edgecolor=edgecolor)
        #
        # plt.legend(scatterpoints=10)
        #
        # self.makeprojectdir("data")
        # plt.xlabel("Radius")
        # plt.ylabel("Percentile")
        # plt.savefig(os.path.join(self.__class__.REPOSITORY_DIR,
        #                         "graph",
        #                         "graph_of_{0}_and_{1}.png".format("Radius", "Percentile")
        #                          )
        #            )
        # plt.clf()
        # 暫定ここまで

        plt.scatter(*straight, color='#FF5A56', alpha=alpha,
                    edgecolor=edgecolor, label="Straight")
        plt.scatter(*brake, color='#CCB947', alpha=alpha,
                    edgecolor=edgecolor, label="Brake")
        plt.scatter(*right, color='#5E6AFF', alpha=alpha,
                    edgecolor=edgecolor, label="Right_LC")
        plt.scatter(*left, color='#3F993E', alpha=alpha,
                    edgecolor=edgecolor, label="Left_LC")

        # CC 紫 #B122B2 #2FCDB4 水色 #FBA848 オレンジ

        # 円を描くとき
        # plt.Circle((0, 0), radius=3, alpha=0)
        
        # 車の順番がおかしい
        #print(right[0], right[1])

        plt.legend(scatterpoints=int(1/alpha))# 薄い時にレジェンドが見えるように数を増やす試み 二乗してもいいか？

        # plt.title("{0} and {1}".format(feature1.value, feature2.value))
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.xlabel("{0}[{1}]".format(feature1.name, feature1.unit()))
        plt.ylabel("{0}[{1}]".format(feature2.name, feature2.unit()))
        self.makeprojectdir("graph")
        plt.savefig(os.path.join(self.__class__.REPOSITORY_DIR,
                                 "graph",
                                 "graph_of_{0}_and_{1}.png".format(feature1.value, feature2.value)
                                 )
                    )
        plt.clf()

    # deprecated
    def get_label_start_sequence(self):
        labels = [lc for dataDict in self.data_dicts for lc in dataDict['roa']]
        sur_rows = [sur_row for sur in [dataDict['sur'] for dataDict in self.data_dicts] for sur_row in sur]
        label_list = []
        #厳密には、開始と終了が繋がる可能性はあるが、まあないだろう。
        # labelのなかで、車線変更開始時点のラベル、または直進ならそのままで、それ以外は-2とか
        b = self.__class__.start_index(labels)
        c = list(b[0])
        c.extend(b[1])
        labels = [a if a == 0 or i in c else a*2 for i, a in enumerate(labels)]
        for label, sur_row in zip(labels, pb.single_generator(sur_rows)):

            cars = self.get_cars(sur_row)
            for _ in cars:
                label_list.append(label)
        return label_list

    # deprecated
    def get_label_sequence(self):

        labels = [lc for dataDict in self.data_dicts for lc in dataDict['roa']]
        sur_rows = [sur_row for sur in [dataDict['sur'] for dataDict in self.data_dicts] for sur_row in sur]
        label_list = []
        for label, sur_row in zip(labels, pb.single_generator(sur_rows)):
            cars = self.get_cars(sur_row)
            for _ in cars:
                label_list.append(label)
        return label_list

    def get_feature_sequence(self, feature):
        sur_rows = [sur_row for sur in [dataDict['sur'] for dataDict in self.data_dicts] for sur_row in sur]
        feature_list = []
        for sur_row in pb.single_generator(sur_rows):
            cars = self.get_cars(sur_row)
            for car in cars:
                feature_list.append(self.calc_feature_from_car(car, feature))
        return feature_list


    def label_subjectdict_brakefiltered(self):
        print("deprecated")
        '''
        オリジナルのラベルデータから、被験者名ごとの辞書にして渡す。
        :return: ？？？
        '''
        # data_dictsのデータ構造を被験者ごとにしたい感じはある。
        label_dict = {}

        # 暫定的にlabelを変更する策に出る
        for name, data_dict in zip(pb.single_generator(self.behaviornames), self.data_dicts):
            label_dict[name] = dividelabelwithbrake(data_dict['roa'], data_dict['drv'][:, 0], threshold=5)
        return label_dict

    # def label_sequence(self):
    #     '''
    #     オリジナルのラベルデータから、被験者名ごとの辞書にして渡す。
    #     :return: ？？？
    #     '''
    #     # data_dictsのデータ構造を被験者ごとにしたい感じはある。
    #     label_dict = {}
    #     
    #     for name, data_dict in zip(self.behavior_names, self.data_dicts):
    #         label_dict[name] = data_dict['roa']
    #     return label_dict

    def toaccel(self, data_dict):
        sur = data_dict['sur']

        l = []
        for i in range(len(sur)):
            if i == 0:
                m = []
                for j in range(int(len(sur[0]) / 4)):
                    m.append([*sur[0][j * 4:j * 4 + 4], 0, 0])
                l.append(m)
            else:
                m = []
                for j in range(int(len(sur[0]) / 4)):
                    m.append([*sur[i][j * 4:j * 4 + 4], sur[i][4 * j + 2] - sur[i - 1][4 * j + 2],
                              sur[i][4 * j + 3] - sur[i - 1][4 * j + 3]])
                l.append(m)
        return l

    def feature_subjectdict(self, *features):
        """
        データ構造
        ある瞬間の周辺車に対する特徴
        in 各フレームの値
        in 運転行動の被験者名がkeyの辞書
        in 特徴数分の配列
        """

        def feature_list_from_data_dict(feature, data_dict):
            feature_list = []
            sur = self.toaccel(data_dict)
            for sur_at_moment, lc_at_moment in zip(sur, data_dict['roa']):
                feature_at_moment = []
                for car in self.get_cars(sur_at_moment):
                    feature_at_moment.append(self.calc_feature_from_car(car, feature))
                feature_list.append(feature_at_moment)
            return feature_list

        feature_dicts = []
        for feature in features:
            feature_dict = {}
            if feature.value == "label":
                # 暫定的にlabelを変更する策に出る
                for name, data_dict in zip(pb.single_generator(self.behaviornames), self.data_dicts):
                    feature_dict[name] = dividelabelwithbrake(data_dict['roa'], data_dict['drv'][:, 0], threshold=5)
            else:
                for data_dict, subjectName in zip(pb.single_generator(self.data_dicts), self.behaviornames):
                    feature_dict[subjectName] = feature_list_from_data_dict(feature, data_dict)

            feature_dicts.append(feature_dict)
        if len(feature_dicts) == 1:
            return feature_dicts[0]
        else:
            return feature_dicts


    def calc_feature_from_car(self, car, feature):
        x = car[0]
        y = car[1]
        vx = car[2]
        vy = car[3]
        ax = car[4]
        ay = car[5]

        def solve_quad(a, b, c):
            if a == 0:
                return - b / c, - b / c
            else :
                return (-b + math.sqrt(pow(b, 2) - 4 * a * c)) / 2 / a, (-b - math.sqrt(pow(b, 2) - 4 * a * c)) / 2 / a


        if feature.value == 'ttcp':
            if vx == 0 and vy == 0:
                return float('inf')
            else:
                return -(x * vx + y * vy) * 1000 / (vx ** 2 + vy ** 2) / 3600  # km/h*1000/3600 = m/s
        elif feature.value == "dtcp":
            if vx == 0 and vy == 0:
                return float('inf')
            else:
                return abs(x * vy - y * vx) / math.sqrt(vx ** 2 + vy ** 2)
        elif feature.value == "ttcx":
            try:
                sols = solve_quad(1/2*ax, vx, (x - self.WIDTH_OF_CARS * np.sign(x)))
                if sols[0] > 0 and sols[1] > 0:
                    return min(sols)* 1000 / 3600
                else:
                    return max(sols)* 1000 / 3600
            except:
                return float('inf')
        elif feature.value == "ttcy":
            try:
                sols = solve_quad(1/2*ay, vy, (y - self.WIDTH_OF_CARS * np.sign(y)))
                if sols[0] > 0 and sols[1] > 0:
                    return min(sols) * 1000 / 3600
                else:
                    return max(sols) * 1000 / 3600
            except:
                return float('inf')
        elif feature.value == "dist":
            return math.sqrt(x**2 + y**2)
        elif feature.value == "deg":
            return math.atan2(y, x)/math.pi*180

    # def show_plot(self, nameOfFeature):
    #     i = self.featureNames.index(nameOfFeature)
    #
    #     flag = True
    #     for j, (label, someFeature) in enumerate(zip(pb.single_generator(self.oneDimVectors), self.twoDimVectors)):
    #         feature = someFeature[:, i]
    #         t = np.arange(0, len(label) / 10, 0.1)
    #
    #         for k, labelIndex in enumerate(self.__class__.get_before_and_after_LC_label(label)):
    #             tmp_t = np.arange(0, len(labelIndex) / 10, 0.1)
    #             # tmp_t = t[labelIndex]
    #             tmp_feature = feature[labelIndex]
    #             tmp_label = label[labelIndex]
    #
    #             right = (tmp_t[np.where(tmp_label == 1)],
    #                      tmp_feature[np.where(tmp_label == 1)])
    #             straight = (tmp_t[np.where(tmp_label == 0)],
    #                         tmp_feature[np.where(tmp_label == 0)])
    #             left = (tmp_t[np.where(tmp_label == -1)],
    #                     tmp_feature[np.where(tmp_label == -1)])
    #
    #             plt.scatter(*straight, color='#B122B2', alpha=0.5,
    #                         edgecolor='none', label="Straight")
    #             plt.scatter(*left, color='#FBA848', alpha=0.5,
    #                         edgecolor='none', label="Left_LC")
    #             plt.scatter(*right, color='#2FCDB4', alpha=0.5,
    #                         edgecolor='none', label="Right_LC")
    #             if flag:
    #                 plt.legend()
    #                 flag = False
    #
    #                 # plt.title("graph Title")
    #                 # plt.xlim([tmp_label[0]/10,tmp_label[len(tmp_label) - 1]]/10)
    #
    #     os.makedirs('graph/', exist_ok=True)
    #     plt.xlabel("Time[sec]")
    #     plt.ylabel(nameOfFeature)
    #     plt.savefig('graph/' + nameOfFeature + '.pdf')
    #     plt.clf()

# read の段階で単位を揃えたい
    
    # ラベルの段階でroaとかじゃなくて特徴名(LCとか)でアクセスできるようにすればいいんじゃないの？
    def read_6000(self):
        DATA_PATH_6000 = self.__class__.DATA_PATH_6000
        dataDicts = []

        tmpList = sorted(os.listdir(DATA_PATH_6000))

        bar = pb(len(tmpList))
        print('6000番台読込中')

        for i, subject in enumerate(tmpList):
            for task in sorted(os.listdir(os.path.join(DATA_PATH_6000, subject))):
                self.behaviornames.append(subject + task)
                drvDF = pd.read_csv(os.path.join(DATA_PATH_6000, subject, task, subject + task + '-HostV_DrvInfo.csv'),
                                    encoding='shift-jis', header=0,
                                    names=['time', 'brake', 'gas', 'vel', 'steer', 'accX', 'accY', 'accZ', 'NaN'], dtype='float16')
                drvDF = drvDF.drop(['time', 'NaN'], axis=1)
                roaDF = pd.read_csv(os.path.join(DATA_PATH_6000, subject, task, subject +
                                                 task + '-HostV_RoadInfo.csv'), encoding='shift-jis', header=0, dtype={'LC':'int8'})
                roaDF = roaDF['LC']
                surDF = pd.read_csv(os.path.join(DATA_PATH_6000, subject, task, subject +
                                                 task + '-SurVehicleInfo.csv'), encoding='shift-jis', header=0, dtype='float16')

                # def rangeindex(self):
                #     return self.set_index([list(range(self.shape[0]))])
                #
                # pd.DataFrame.rangeindex = rangeindex
                #
                # dataDF = pd.concat([drvDF, roaDF, surDF], axis=1).dropna()
                # dataDF = dataDF.rangeindex()

                # print(dataDF.shape)
                # print(np.where(dataDF.isnull().any(axis=1)))

                dataDict = {'drv': drvDF.as_matrix(
                ), 'roa': roaDF.as_matrix(), 'sur': surDF.as_matrix()}
                dataDicts.append(dataDict)
            #pandastameshi
            bar.display_progressbar(i)
        return dataDicts

    def read_9000(self):
        DATA_PATH_9000 = self.__class__.DATA_PATH_9000
        dataDicts = []

        print('9000番台読込中')
        bar = pb(sorted(os.listdir(DATA_PATH_9000)))
        for i, item in enumerate(bar.generator(0)):
            for j, data in enumerate(sorted(os.listdir(os.path.join(DATA_PATH_9000, item)))):
                print(data)
                if i == 0:
                    self.behaviornames.append(data)
                    drvDF = pd.read_csv(os.path.join(
                        DATA_PATH_9000, item, data), encoding='shift-jis', header=0, dtype='float16')
                    drvDF = drvDF.drop(
                        ['time[sec]', 'lat[deg]', 'lon[deg]'], axis=1)
                    drvDF = drvDF.dropna()
                    dataDicts.append({'drv': drvDF.as_matrix()})
                elif i == 1:
                    roaDF = pd.read_csv(os.path.join(
                        DATA_PATH_9000, item, data), encoding='shift-jis', header=0, dtype={'LC':'int8'})
                    roaDF = roaDF['LC']
                    dataDicts[j]['roa'] = roaDF.as_matrix()
                elif i == 2:
                    surDF = pd.read_csv(os.path.join(
                        DATA_PATH_9000, item, data), encoding='shift-jis', header=0, dtype='float16')
                    dataDicts[j]['sur'] = surDF.as_matrix()

        return dataDicts

    def save_dataDicts(self):
        self.makeprojectdir("data")
        np.save(os.path.join(self.__class__.REPOSITORY_DIR, "data", "dataDicts.npy"), self.data_dicts)
        np.save(os.path.join(self.__class__.REPOSITORY_DIR, "data", "behaviornames.npy"), self.behaviornames)

    def save_vectors(self):
        self.makeprojectdir("data")
        np.save(os.path.join(self.__class__.REPOSITORY_DIR, "data", "twoDimVectors.npy"), self.twoDimVectors)
        np.save(os.path.join(self.__class__.REPOSITORY_DIR, "data", "oneDimVectors.npy"), self.oneDimVectors)
        np.save(os.path.join(self.__class__.REPOSITORY_DIR, "data", "featureNames.npy"), self.featureNames)

    def save_label_and_feature(self):
        self.makeprojectdir("data")
        np.save(os.path.join(self.__class__.REPOSITORY_DIR, "data", "feature.npy"), self.feature)
        np.save(os.path.join(self.__class__.REPOSITORY_DIR, "data", "label.npy"), self.label)

    def adjust_size(self):
        for dataDict in self.data_dicts:
            drv = dataDict['drv']
            roa = dataDict['roa']
            sur = dataDict['sur']
            # debug print
            print(drv.shape[0], roa.shape[0], sur.shape[0])

            min_size = min(drv.shape[0], roa.shape[0], sur.shape[0])
            diffs = (min_size - drv.shape[0], min_size -
                     roa.shape[0], min_size - sur.shape[0])
            if diffs[0] != 0:
                drv = drv[:diffs[0], :]

            if diffs[1] != 0:
                roa = roa[:diffs[1]]  # ,:] roadを他にも使うなら、一列じゃなくなるなら

            if diffs[2] != 0:
                sur = sur[:diffs[2], :]

            # こちらだと、enumerableにして[i]を付けないと更新されない。おそらく新しいオブジェクトになってしまうから？よくわかんね
            # ✕ dataDict = {'drv':drv,'roa':roa,'sur':sur}
            # ◯ self.dataDict[i] = {'drv':drv,'roa':roa,'sur':sur}
            dataDict['drv'] = drv
            dataDict['roa'] = roa
            dataDict['sur'] = sur

    def is_all_same_size(self):
        for dataDict in self.data_dicts:
            if not len(dataDict['drv']) == len(dataDict['roa']) == len(dataDict['sur']):
                return False
        return True

    def assign_lc_to_oneDimVectors(self):
        self.oneDimVectors = [dataDict['roa'] for dataDict in self.data_dicts]

    def add_drv_to_twoDimVectors(self):
        self.concat_twoDimVectors([dataDict['drv'] for dataDict in self.data_dicts],
                                  ['Brake[N]', 'Gas Pedal[N]', 'Velocity[km/h]',
                                   'Steering angle[deg]', 'Acceleration_X[G]', 'Acceleration_Y[G]',
                                   'Acceleration_Z[G]'])  # ここの名称の自動化したい

    def add_ttc_to_twoDimVectors(self):
        # ttcs = np.load('ttcs.npy')
        # self.concat_twoDimVectors(ttcs,["TTC" + str(i) for i in range(54)])
        # return

        WIDTH_OF_CARS = self.__class__.WIDTH_OF_CARS
        LENGTH_OF_CARS = self.__class__.LENGTH_OF_CARS

        ttcs = []
        surs = [dataDict['sur'] for dataDict in self.data_dicts]

        print('ttc計算中')
        for sur in pb.single_generator(surs):
            ttc = []
            for sur_row in sur:

                cars = self.get_cars(sur_row)

                ttcXBlock = NeighborBlock()
                ttcYBlock = NeighborBlock()

                for car in cars:
                    if np.isnan((car[0] - WIDTH_OF_CARS * np.sign(car[0])) / car[2]):
                        ttcXBlock.add(float('inf'), car[0], car[1])
                    else:
                        # すでにx方向には衝突してる時とかの処理
                        ttcXBlock.add(
                            (car[0] - WIDTH_OF_CARS * np.sign(car[0])) / car[2], car[0], car[1])

                    if np.isnan((car[1] - LENGTH_OF_CARS * np.sign(car[1])) / car[3]):
                        ttcYBlock.add(float('inf'), car[0], car[1])
                    else:
                        ttcYBlock.add(
                            (car[1] - LENGTH_OF_CARS * np.sign(car[1])) / car[3], car[0], car[1])

                ttc_row = ttcXBlock.get_list_atan()
                ttc_row.extend(ttcYBlock.get_list_atan())

                ttc.append(ttc_row)
            ttcs.append(np.array(ttc))
        # 改修 処理かぶりまくり　
        featureNames = ['TTC_{0} {1}_{2}'.format(axis, Dist(
            i // 3).name, Side(i % 3).name) for axis in ('X', 'Y') for i in range(27)]
        self.concat_twoDimVectors(ttcs, featureNames)
        # np.save('ttcs.npy', ttcs)

    def add_ttn_to_twoDimVectors(self):
        # ttcs = np.load('ttns.npy')
        # self.concat_twoDimVectors(ttcs,["TTN" + str(i) for i in range(27)])
        # return

        WIDTH_OF_CARS = self.__class__.WIDTH_OF_CARS
        LENGTH_OF_CARS = self.__class__.LENGTH_OF_CARS

        ttns = []
        print('ttn計算中')
        # おんなじ処理をしてるしクソ

        surs = [dataDict['sur'] for dataDict in self.data_dicts]

        for sur in pb.single_generator(surs):
            ttn = []

            for sur_row in sur:

                cars = self.get_cars(sur_row)

                ttnBlock = NeighborBlock()
                for car in cars:
                    ttnBlock.add(self.calc_ttn(car), car[0], car[1])

                ttn_row = ttnBlock.get_list_atan()

                ttn.append(ttn_row)
            ttns.append(np.array(ttn))

        featureNames = ['TTN {0}_{1}'.format(
            Dist(i // 3).name, Side(i % 3).name) for i in range(27)]

        self.concat_twoDimVectors(ttns, featureNames)
        # np.save('ttns.npy', ttns)

    def add_distAndVel_to_twoDimVectors(self):
        distAndVels = []
        print('距離と速度を特徴にしています')

        surs = [dataDict['sur'] for dataDict in self.data_dicts]
        initArrayx = [[float('-inf'), float('inf'), float('inf')]
                      for _ in range(9)]
        initArrayy = [float('inf') for _ in range(15)]
        initArrayy.extend([float('-inf') for _ in range(12)])

        for sur in pb.single_generator(surs):
            distAndVel = []
            for sur_row in sur:
                cars = self.get_cars(sur_row)

                xBlock = NeighborBlock(initArrayx)
                yBlock = NeighborBlock(initArrayy)
                vxBlock = NeighborBlock(initArrayx)
                vyBlock = NeighborBlock(initArrayy)
                for car in cars:
                    xBlock.add(car[Sub.x0], car[0], car[1])
                    yBlock.add(car[Sub.y0], car[0], car[1])
                    vxBlock.add(car[Sub.vx], car[0], car[1])
                    vyBlock.add(car[Sub.vy], car[0], car[1])

                distAndVel_row = xBlock.get_list_atan()
                distAndVel_row.extend(yBlock.get_list_atan())
                distAndVel_row.extend(vxBlock.get_list_atan())
                distAndVel_row.extend(vyBlock.get_list_atan())

                distAndVel.append(distAndVel_row)
            distAndVels.append(np.array(distAndVel))

        featureNames = ['{0} {1} {2}'.format(kind, Dist(
            i // 3).name, Side(i % 3).name) for kind in ['x', 'y', 'vx', 'vy'] for i in range(27)]
        self.concat_twoDimVectors(distAndVels, featureNames)
        # np.save('dinve.npy', distAndVels)

    def add_ttcpanddtcp_to_twoDimVectors(self):

        ttcps = []
        dtcps = []
        print('ttcp計算中')
        # おんなじ処理をしてるしクソ

        surs = [dataDict['sur'] for dataDict in self.data_dicts]

        for sur in pb.single_generator(surs):
            ttcp = []
            dtcp = []

            for sur_row in sur:

                cars = self.get_cars(sur_row)

                ttcpBlock = NeighborBlock()
                dtcpBlock = NeighborBlock()
                for car in cars:
                    ttcpBlock.add(self.calc_feature_from_car(car, Features.TimeToClosestPoint), car[0], car[1])
                    dtcpBlock.add(self.calc_feature_from_car(car, Features.DistanceToClosestPoint), car[0], car[1])

                ttcp_row = ttcpBlock.get_list_atan()
                dtcp_row = dtcpBlock.get_list()

                ttcp.append(ttcp_row)
                dtcp.append(dtcp_row)
            ttcps.append(np.array(ttcp))
            dtcps.append(np.array(dtcp))

        featureNames = ['{0} {1} {2}'.format(kind, Dist(
            i // 3).name, Side(i % 3).name) for kind in ['TTCP', 'DTCP'] for i in range(27)]
        ttcps.append(np.array(dtcps))
        self.concat_twoDimVectors(ttcps, featureNames)
        # np.save('ttns.npy', ttns)

    def get_cars(self, sur_row):
        cars= np.array(sur_row)
        # cars = sur_row.reshape(int(sur_row.shape[0] / 6), 6)

        cars_i = []
        for i, car in enumerate(cars):
            if car[0] == car[1] == car[2] == car[3] == 0:
                cars_i.append(i)

        # もうちょっとやりようはあったと思うが・・・
        # deleteじゃなくても引数に指定インデックス入れれば取得できたはず。
        cars = np.delete(cars, cars_i, 0)

        dist_list = [np.linalg.norm([car[0], car[1]]) for car in cars]
        sort_i = np.argsort(dist_list)  # [:8]

        return cars[sort_i]

    def calc_ttn(self, car):
        WIDTH_OF_CARS = self.__class__.WIDTH_OF_CARS
        LENGTH_OF_CARS = self.__class__.LENGTH_OF_CARS

        var('t', positive=True, real=True)

        x0 = car[Sub.x0]
        y0 = car[Sub.y0]
        vx = car[Sub.vx]
        vy = car[Sub.vy]

        x = x0 + vx * t  # + 1/2*ax*t**2
        y = y0 + vy * t  # + 1/2*ay*t**2

        def divide(numer, denom):
            if denom == 0:
                return float('inf')
            else:
                return numer / denom

        def sides_condition(t_c):
            return abs(divide(y0 + vy * t_c, x0 + vx * t_c)) <= LENGTH_OF_CARS / WIDTH_OF_CARS

        def froRea_condition(t_c):
            return abs(divide(y0 + vy * t_c, x0 + vx * t_c)) > LENGTH_OF_CARS / WIDTH_OF_CARS

        solutions = []
        try:
            solutionSides = divide(np.sign(x0) * WIDTH_OF_CARS - x0, vx)
            if sides_condition(solutionSides):
                solutions.append(solutionSides)
        except RuntimeError:
            pass

        try:
            solutionFroRea = divide(np.sign(y0) * LENGTH_OF_CARS - y0, vy)
            if froRea_condition(solutionFroRea):
                solutions.append(solutionFroRea)
        except RuntimeError:
            pass

        solutions.append(float('inf'))
        return min([sol for sol in solutions if sol > 0])

    def write_log(self, *texts):
        texts = [str(text) for text in texts]
        import csv
        with open('log.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(texts)

    def concat_twoDimVectors(self, newTwoDimVectors, featureNames):
        self.featureNames.extend(featureNames)
        if not len(self.twoDimVectors):
            self.twoDimVectors = newTwoDimVectors
        else:
            if not len(newTwoDimVectors) == len(self.twoDimVectors):
                print("運転行動数が一致していません。スクリプトがおかしい可能性大")
            for i, (oldTwoDimVector, newTwoDimVector) in enumerate(zip(self.twoDimVectors, newTwoDimVectors)):
                self.twoDimVectors[i] = np.c_[oldTwoDimVector, newTwoDimVector]

    def retain_sequence(self, numOfSequences):
        for i, (oneDimVector, twoDimVector) in enumerate(
                zip(pb.single_generator(self.oneDimVectors), self.twoDimVectors)):
            tdv = twoDimVector
            p = numOfSequences * tdv.shape[1]
            n = tdv.shape[0] - numOfSequences + 1

            sequence = []
            for j in range(n):
                tmp = twoDimVector[j:numOfSequences + j, :]
                sequence.append(tmp[::-1, :].reshape(p))  # 配列の上下反転::-1

            self.oneDimVectors[i] = oneDimVector[numOfSequences - 1:]
            self.twoDimVectors[i] = np.array(sequence)
        print(str(numOfSequences) + "フレーム保持しました")

    def map_all_m1_to_1(self):
        for i, oneDimVector in enumerate(self.oneDimVectors):
            self.oneDimVectors[i] = list(
                map(lambda label: 1 if label == -1 else label, oneDimVector))
        print("-1のラベルを1へ変更しました。")

    def delete_3frames_after(self):
        for i, (oneDimVector, twoDimVector) in enumerate(zip(self.oneDimVectors, self.twoDimVectors)):
            rightLcLabel = 0
            leftLcLabel = 0
            delIndex = []
            for j, label in enumerate(oneDimVector):
                if label == 0:
                    rightLcLabel = 0
                    leftLcLabel = 0
                elif label == 1:
                    rightLcLabel += 1
                    if rightLcLabel > 3:
                        leftLcLabel = 0
                        delIndex.append(j)
                elif label == -1:
                    leftLcLabel += 1
                    if leftLcLabel > 3:
                        rightLcLabel = 0
                        delIndex.append(j)
                else:
                    print('lcに0,1,-1の文字が含まれています。')

            lcUseIndex = list(set(range(len(oneDimVector))) - set(delIndex))
            self.oneDimVectors[i] = np.delete(oneDimVector, delIndex)
            self.twoDimVectors[i] = np.delete(twoDimVector, delIndex, 0)
        print("3フレーム以降切り捨てました。")

    def vectors_to_label_and_feature(self):
        self.label = np.array(
            [label for oneDimVector in self.oneDimVectors for label in oneDimVector])
        self.feature = f = np.array(
            [feature for twoDimVector in self.twoDimVectors for feature in twoDimVector])

    def connect_label_and_feature(self, predFrames):
        self.label = self.label[predFrames:]
        self.feature = self.feature[:-1 * predFrames]

    def is_label_and_feature_same_size(self):
        return self.label.shape[0] == self.feature.shape[0]

    def __init__(self, dataInput):
        self.label = np.array([])
        self.feature = np.array([])
        self.twoDimVectors = []
        self.oneDimVectors = None
        self.featureNames = []
        self.behaviornames = []
        if dataInput.value == 'read':
            d = []
            d.extend(self.read_6000())
            d.extend(self.read_9000())
            self.data_dicts = d
        elif dataInput.value == 'origin':
            datadicts_path = os.path.join(self.__class__.REPOSITORY_DIR, "data", "dataDicts.npy")
            behaviornames_path = os.path.join(self.__class__.REPOSITORY_DIR, "data", "behaviornames.npy")
            self.data_dicts = np.load(datadicts_path)
            self.behaviornames = np.load(behaviornames_path)
            print("読み込みが完了しました。")
        elif dataInput.value == 'vector':
            np.save(os.path.join(self.__class__.REPOSITORY_DIR, "data", "dataDicts.npy"), self.data_dicts)

            self.oneDimVectors = np.load(os.path.join(self.__class__.REPOSITORY_DIR, "data", "oneDimVectors.npy"))
            self.twoDimVectors = np.load(os.path.join(self.__class__.REPOSITORY_DIR, "data", "twoDimVectors.npy"))
            self.featureNames = np.load(os.path.join(self.__class__.REPOSITORY_DIR, "data", "featureNames.npy"))
        else:
            print("初期化に失敗しました。第一引数には列挙体DataInputの値を入力してください")

if __name__ == '__main__':
    print("lane_changing utility")
