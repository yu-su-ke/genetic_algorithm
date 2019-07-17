import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from pymongo import MongoClient
import re
import distance_calculation


class GeneticAlgorithm:
    # Initialization
    def __init__(self, path=None, n_gene=256, n_parent=10, change_ratio=0.1):
        self.n_gene = n_gene  # 一世代の遺伝子の個数
        self.n_parent = 10  # 親として残す個体数
        self.change_ratio = change_ratio  # 突然変異で変化させる場所の数
        self.distance = distance_calculation.DistCalculation()
        self.before_each_distance = []
        self.after_each_distance = []
        self.all_distance = []
        if path is not None:
            self.set_loc(np.array(pd.read_csv(path)))

    # Initialize the gene randomly
    def init_genes(self, ):
        self.genes = np.zeros((self.n_gene, self.num_data), np.int)
        order = np.arange(self.num_data)
        for i in range(self.n_gene):
            np.random.shuffle(order)
            self.genes[i] = order.copy()
        self.sort()

    # Set the coordinates
    def set_location(self, locations):
        self.loc = locations  # x,y座標
        self.num_data = len(self.loc)  # データ数
        self.dist = distance.squareform(distance.pdist(self.loc))  # 距離の表を作成
        self.init_genes()  # 遺伝子を初期化

    def cost(self, order):
        return np.sum([self.dist[order[i], order[(i + 1) % self.num_data]] for i in np.arange(self.num_data)])

    def plot(self, country_list, order=None):
        # 初期配置
        if order is None:
            for i in range(len(self.loc[:, 1]) - 1):
                country1 = self.loc[:, 1][i], self.loc[:, 0][i]
                country2 = self.loc[:, 1][i+1], self.loc[:, 0][i+1]
                print(country1, country2)
                # print(dist.dist_on_sphere(country1, country2))
                # before_each_distance.append(dist.dist_on_sphere(country1, country2))
                print(dist.dist_test(country1, country2))
                self.before_each_distance.append(dist.dist_test(country1, country2))
            plt.plot(self.loc[:, 0], self.loc[:, 1])
            plt.plot(self.loc[:, 0], self.loc[:, 1], 'o', markersize=6)
            for i, (x, y) in enumerate(zip(self.loc[:, 0], self.loc[:, 1])):
                plt.annotate(country_list[i], (x, y))

        # 最適化した配置
        else:
            for i in range(len(self.loc[order, 1]) - 1):
                country3 = self.loc[order, 1][i], self.loc[order, 0][i]
                country4 = self.loc[order, 1][i+1], self.loc[order, 0][i+1]
                # print(dist.dist_on_sphere(country1, country2))
                self.after_each_distance.append(self.distance.dist_on_sphere(country3, country4))

            plt.plot(self.loc[order, 0], self.loc[order, 1])
            plt.plot(self.loc[order, 0], self.loc[order, 1], 'o', markersize=6)
            for i, (x, y) in enumerate(zip(self.loc[:, 0], self.loc[:, 1])):
                plt.annotate(country_list[i], (x, y))

        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.show()

        # print('最適化実行前 : ' + str(sum(before_each_distance)))
        print('最適化実行後 : ' + str(sum(self.after_each_distance)))
        self.all_distance.append(round(sum(self.after_each_distance)))

    # Genetic Algorithm
    def gen_algo(self, n_step):
        for i in range(n_step):
            print("Generation : %d, Cost : %lf" % (i, self.cost(self.genes[0])))
            self.evolution()
        self.result = self.genes[0]

        return self.result

    # Genetic evolution
    def evolution(self):
        # 突然変異
        for i in range(self.n_parent, self.n_gene):
            self.genes[i] = self.mutation(np.random.randint(self.n_parent))
        self.sort()  # ソートする

    def sort(self):
        # コストを計算し，ソート
        gene_cost = np.array([self.cost(i) for i in self.genes])
        self.genes = self.genes[np.argsort(gene_cost)]

    # Return the mutated gene
    def mutation(self, index):
        n_change = int(self.change_ratio * self.num_data)
        gene = self.genes[index].copy()

        for i in range(n_change):
            # n_changeの個数だけ値を入れ替える
            left = np.random.randint(self.num_data)
            right = np.random.randint(self.num_data)

            temp = gene[left]
            gene[left] = gene[right]
            gene[right] = temp

        return gene

    def plot_result(self):
        x = [i for i in range(100, 2100, 100)]
        plt.plot(x, self.all_distance)
        plt.xticks(np.arange(100, 2100, 200))
        for (i, j, k) in zip(x, self.all_distance, self.all_distance):
            plt.plot(i, j, 'o')
            plt.annotate(k, xy=(i, j))
        plt.show()


def save_mongo(collection, db):
    save = db[collection]

    with open('./country.txt', 'r', encoding='utf-8') as text:
        for line in text:
            country_list = {}
            line = re.sub('[\r\n]+$', '', line)
            element = line.split(',')
            country_list['country'] = element[0]
            country_list['ido'] = element[1]
            country_list['keido'] = element[2]
            # コレクションに追加(コレクションもここで作成)
            save.insert(country_list)


def make_matrix(collection, db, country_list):
    # print(np.random.random_sample((10, 2)) * 10)

    query = db[collection].find().limit(1000000)
    country_array = np.empty((0, 2), int)

    for record in query:
        country_array = np.append(country_array, np.array([[int(record['keido']), int(record['ido'])]]), axis=0)
        country_list.append(record['country'])
    return country_array, country_list


