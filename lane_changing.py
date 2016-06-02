#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import *

from progressbar import ProgressBar as pb

import matplotlib
import numpy as np
import os
import pandas as pd
from sympy import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DataInput(Enum):
    loadVectorData = 'vector'
    loadOriginalData = 'origin'
    readFromCSVData = 'read'


# クラス内クラスとかできるのか。バウンドインナークラス？


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

    def get_list(self):
        return [feature for feature in self.featureList.reshape(27)]

    def get_list_atan(self):
        return [math.atan(feature) for feature in self.featureList.reshape(27)]

# メソッドを呼ぶ順番を考えなきゃいけないクラスってどうなん？

class Container:
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_PATH_6000 = os.path.join(SCRIPT_DIR, 'data/Original/6000/')
    DATA_PATH_9000 = os.path.join(SCRIPT_DIR, 'data/Original/9000/')

    WIDTH_OF_CARS = 2
    LENGTH_OF_CARS = 4

    def get_start_of_LC_label(label):

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
        # plt.title("Graph Title")
        # plt.xlim([tmp_label[0]/10,tmp_label[len(tmp_label) - 1]]/10)

        le = []
        for j, (label, someFeature) in enumerate(zip(pb.single_generator(self.oneDimVectors), self.twoDimVectors)):
            feature = someFeature[:, i]
            tmp = self.__class__.get_start_of_LC_label(label)[0]
            if len(tmp) != 0:
                le.extend(feature[tmp])

        le = np.delete(np.array(le), np.where(np.array(le) == float('inf')))
        le = np.delete(np.array(le), np.where(np.array(le) == float('-inf')))
        plt.hist(le)

        os.makedirs('Graph/', exist_ok=True)
        plt.xlabel("Value")
        plt.ylabel(nameOfFeature)
        plt.savefig('Graph/' + nameOfFeature + "_r" + '.pdf')
        plt.clf()

        plt.legend()
        # plt.title("Graph Title")
        # plt.xlim([tmp_label[0]/10,tmp_label[len(tmp_label) - 1]]/10)

        le = []
        for j, (label, someFeature) in enumerate(zip(pb.single_generator(self.oneDimVectors), self.twoDimVectors)):
            feature = someFeature[:, i]
            tmp = self.__class__.get_start_of_LC_label(label)[1]
            if len(tmp) != 0:
                le.extend(feature[tmp])

        le = np.delete(np.array(le), np.where(np.array(le) == float('inf')))
        le = np.delete(np.array(le), np.where(np.array(le) == float('-inf')))

        plt.hist(le)

        # ファイル名周り未検証
        os.makedirs(os.path.join(self.__class__.SCRIPT_DIR, 'Graph/'), exist_ok=True)

        plt.xlabel("Value")
        plt.ylabel(nameOfFeature)
        plt.savefig(os.path.join(self.__class__.SCRIPT_DIR,
                                 "Graph",
                                 nameOfFeature.replace('\\', "{BSlash}") + "_l" + '.pdf')
                    )
        plt.clf()

    def show_dtcp_ttcp(self, load=False):

        if load:
            element_of_graph = np.load(os.path.join(self.__class__.SCRIPT_DIR, "tmp", "element_of_graph.npz"))
            time_list = element_of_graph['time_list']
            distance_list = element_of_graph['distance_list']
            label_list = element_of_graph['label_list']
        else:
            time_list, distance_list, label_list = self.get_cars_with_label()
            time_list = np.array(list(map(lambda x: x * 1000 / 3600, time_list)))  # km/h*1000/3600 = m/s
            distance_list = np.array(distance_list)
            label_list = np.array(label_list)

        os.makedirs(os.path.join(self.__class__.SCRIPT_DIR, "tmp"), exist_ok=True)
        np.savez(os.path.join(self.__class__.SCRIPT_DIR, "tmp", "element_of_graph.npz"),
                 time_list=time_list,
                 distance_list=distance_list,
                 label_list=label_list,
                 )

        left = (time_list[np.where(label_list == -1)],
                distance_list[np.where(label_list == -1)])
        straight = (time_list[np.where(label_list == 0)],
                    distance_list[np.where(label_list == 0)])
        right = (time_list[np.where(label_list == 1)],
                 distance_list[np.where(label_list == 1)])
        alpha = 0.05
        edgecolor = 'none'

        plt.scatter(*straight, color='#B122B2', alpha=alpha,
                    edgecolor=edgecolor, label="Straight")
        plt.scatter(*right, color='#2FCDB4', alpha=alpha,
                    edgecolor=edgecolor, label="Right_LC")
        plt.scatter(*left, color='#FBA848', alpha=alpha,
                    edgecolor=edgecolor, label="Left_LC")

        # この解決策はどうなんだ
        plt.legend(scatterpoints=100)

        plt.title("Time To CP and Distance To CP")
        plt.xlim(-5, 5)
        os.makedirs(os.path.join(self.__class__.SCRIPT_DIR, "Graph/"), exist_ok=True)
        plt.xlabel("TimeToClosestPoint[sec]")
        plt.ylabel("DistanceToClosestPoint[m]")
        plt.savefig(os.path.join(self.__class__.SCRIPT_DIR, "Graph", "TTCP_and_DTCP.png"))
        plt.clf()

    def get_cars_with_label(self):
        timeList = []
        distanceList = []
        labelList = []
        labels = [lc for dataDict in self.dataDicts for lc in dataDict['roa']]
        sur_rows = [sur_row for sur in [dataDict['sur'] for dataDict in self.dataDicts] for sur_row in sur]

        for label, sur_row in zip(labels, pb.single_generator(sur_rows)):
            cars = self.get_cars(sur_row)
            for car in cars:
                result = self.calc_timeToClosestPoint(car)
                timeList.append(result[0])
                distanceList.append(result[1])
                labelList.append(label)

        return timeList, distanceList, labelList

    def calc_timeToClosestPoint(self, car):
        x = car[0]
        y = car[1]
        vx = car[2]
        vy = car[3]

        if vx == 0 and vy == 0:
            return float('inf'), float('inf')

        import math
        timeToClosestPoint = -(x * vx + y * vy) / (vx ** 2 + vy ** 2)
        distanceToClosestPoint = abs(x * vy - y * vx) / math.sqrt(vx ** 2 + vy ** 2)

        return timeToClosestPoint, distanceToClosestPoint

    def show_plot(self, nameOfFeature):
        i = self.featureNames.index(nameOfFeature)

        flag = True
        for j, (label, someFeature) in enumerate(zip(pb.single_generator(self.oneDimVectors), self.twoDimVectors)):
            feature = someFeature[:, i]
            t = np.arange(0, len(label) / 10, 0.1)

            for k, labelIndex in enumerate(self.__class__.get_before_and_after_LC_label(label)):
                tmp_t = np.arange(0, len(labelIndex) / 10, 0.1)
                # tmp_t = t[labelIndex]
                tmp_feature = feature[labelIndex]
                tmp_label = label[labelIndex]

                right = (tmp_t[np.where(tmp_label == 1)],
                         tmp_feature[np.where(tmp_label == 1)])
                straight = (tmp_t[np.where(tmp_label == 0)],
                            tmp_feature[np.where(tmp_label == 0)])
                left = (tmp_t[np.where(tmp_label == -1)],
                        tmp_feature[np.where(tmp_label == -1)])

                plt.scatter(*straight, color='#B122B2', alpha=0.5,
                            edgecolor='none', label="Straight")
                plt.scatter(*left, color='#FBA848', alpha=0.5,
                            edgecolor='none', label="Left_LC")
                plt.scatter(*right, color='#2FCDB4', alpha=0.5,
                            edgecolor='none', label="Right_LC")
                if flag:
                    plt.legend()
                    flag = False

                    # plt.title("Graph Title")
                    # plt.xlim([tmp_label[0]/10,tmp_label[len(tmp_label) - 1]]/10)

        os.makedirs('Graph/', exist_ok=True)
        plt.xlabel("Time[sec]")
        plt.ylabel(nameOfFeature)
        plt.savefig('Graph/' + nameOfFeature + '.pdf')
        plt.clf()

    def read_6000(self):
        DATA_PATH_6000 = self.__class__.DATA_PATH_6000
        dataDicts = []

        tmpList = sorted(os.listdir(DATA_PATH_6000))

        bar = pb(len(tmpList))
        print('6000番台読込中')

        for i, subject in enumerate(tmpList):
            for task in sorted(os.listdir(os.path.join(DATA_PATH_6000, subject))):
                self.subjectNames.append(subject + task)
                drvDF = pd.read_csv(os.path.join(DATA_PATH_6000, subject, task, subject + task + '-HostV_DrvInfo.csv'),
                                    encoding='shift-jis', header=0,
                                    names=['time', 'brake', 'gas', 'vel', 'steer', 'accX', 'accY', 'accZ', 'NaN'])
                drvDF = drvDF.drop(['time', 'NaN'], axis=1)
                # drvDF = drvDF.fillna(method='ffill')#暫定処置。
                roaDF = pd.read_csv(os.path.join(DATA_PATH_6000, subject, task, subject +
                                                 task + '-HostV_RoadInfo.csv'), encoding='shift-jis', header=0)
                roaDF = roaDF['LC']
                surDF = pd.read_csv(os.path.join(DATA_PATH_6000, subject, task, subject +
                                                 task + '-SurVehicleInfo.csv'), encoding='shift-jis', header=0)
                dataDict = {'drv': drvDF.as_matrix(
                ), 'roa': roaDF.as_matrix(), 'sur': surDF.as_matrix()}
                dataDicts.append(dataDict)

            bar.display_progressbar(i)

        return dataDicts

    # 0,1,2がdrv,road,surになってる保証なしだけど、アルファベット順のおかげかうまいこといっている。
    def read_9000(self):
        DATA_PATH_9000 = self.__class__.DATA_PATH_9000
        dataDicts = []

        print('9000番台読込中')
        bar = pb(os.listdir(DATA_PATH_9000))
        for i, item in enumerate(bar.generator(0)):
            for j, data in enumerate(sorted(os.listdir(os.path.join(DATA_PATH_9000, item)))):
                if i == 0:
                    self.subjectNames.append(data)
                    drvDF = pd.read_csv(os.path.join(
                        DATA_PATH_9000, item, data), encoding='shift-jis', header=0)
                    drvDF = drvDF.drop(
                        ['time[sec]', 'lat[deg]', 'lon[deg]'], axis=1)
                    drvDF = drvDF.dropna()
                    dataDicts.append({'drv': drvDF.as_matrix()})
                elif i == 1:
                    roaDF = pd.read_csv(os.path.join(
                        DATA_PATH_9000, item, data), encoding='shift-jis', header=0)
                    roaDF = roaDF['LC']
                    dataDicts[j]['roa'] = roaDF.as_matrix()
                elif i == 2:
                    surDF = pd.read_csv(os.path.join(
                        DATA_PATH_9000, item, data), encoding='shift-jis', header=0)
                    dataDicts[j]['sur'] = surDF.as_matrix()

        return dataDicts

    def save_dataDicts(self):
        os.mkdir('data', exist_ok=True)
        np.save(os.path.join(self.__class__.SCRIPT_DIR, "data", "dataDicts.npy"), self.dataDicts)
        np.save(os.path.join(self.__class__.SCRIPT_DIR, "data", "subjectNames.npy"), self.subjectNames)

    def save_vectors(self):
        os.mkdir('data', exist_ok=True)
        np.save(os.path.join(self.__class__.SCRIPT_DIR, "data", "twoDimVectors.npy"), self.twoDimVectors)
        np.save(os.path.join(self.__class__.SCRIPT_DIR, "data", "oneDimVectors.npy"), self.oneDimVectors)
        np.save(os.path.join(self.__class__.SCRIPT_DIR, "data", "featureNames.npy"), self.featureNames)

    def save_label_and_feature(self):
        os.mkdir('data', exist_ok=True)
        np.save(os.path.join(self.__class__.SCRIPT_DIR, "data", "feature.npy"), self.feature)
        np.save(os.path.join(self.__class__.SCRIPT_DIR, "data", "label.npy"), self.label)

    def adjust_size(self):
        for dataDict in self.dataDicts:
            drv = dataDict['drv']
            roa = dataDict['roa']
            sur = dataDict['sur']

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
        for dataDict in self.dataDicts:
            if not len(dataDict['drv']) == len(dataDict['roa']) == len(dataDict['sur']):
                return False
        return True

    def assign_lc_to_oneDimVectors(self):
        self.oneDimVectors = [dataDict['roa'] for dataDict in self.dataDicts]

    def add_drv_to_twoDimVectors(self):
        self.concat_twoDimVectors([dataDict['drv'] for dataDict in self.dataDicts],
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
        surs = [dataDict['sur'] for dataDict in self.dataDicts]

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

        surs = [dataDict['sur'] for dataDict in self.dataDicts]

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

        surs = [dataDict['sur'] for dataDict in self.dataDicts]
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

    def get_cars(self, sur_row):
        cars = sur_row.reshape(int(sur_row.shape[0] / 4), 4)

        cars_i = []
        for i, car in enumerate(cars):
            if car[0] == car[1] == car[2] == car[3] == 0:
                cars_i.append(i)

        # もうちょっとやりようはあったと思うが・・・
        # deleteじゃなくても引数に指定インデックス入れれば取得できたはず。
        cars = np.delete(cars, cars_i, 0)

        dist_list = [np.linalg.norm([car[0], car[1]]) for car in cars]
        sort_i = np.argsort(dist_list)[:8]

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
        self.subjectNames = []
        if dataInput.value == 'read':
            d = []
            d.extend(self.read_6000())
            d.extend(self.read_9000())
            self.dataDicts = d
        elif dataInput.value == 'origin':
            self.dataDicts = np.load(os.path.join(self.__class__.SCRIPT_DIR, "data", "dataDicts.npy"))
        elif dataInput.value == 'vector':
            np.save(os.path.join(self.__class__.SCRIPT_DIR, "data", "dataDicts.npy"), self.dataDicts)

            self.oneDimVectors = np.load(os.path.join(self.__class__.SCRIPT_DIR, "data", "oneDimVectors.npy"))
            self.twoDimVectors = np.load(os.path.join(self.__class__.SCRIPT_DIR, "data", "twoDimVectors.npy"))
            self.featureNames = np.load(os.path.join(self.__class__.SCRIPT_DIR, "data", "featureNames.npy"))
        else:
            print("初期化に失敗しました。第一引数には列挙体DataInputの値を入力してください")
