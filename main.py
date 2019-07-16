from pymongo import MongoClient
import genetic_algorithm


ga = genetic_algorithm.GeneticAlgorithm()
client = MongoClient('localhost', 27017)
collection = 'country'
db = client.mydb
# genetic_algorithm.save_mongo(collection, db)
country_list = []

for i in range(100, 2100, 100):
    country_location, country_list = genetic_algorithm.make_matrix(collection, db, country_list)
    ga.set_location(country_location)
    ga.gen_algo(n_step=i)
    ga.plot(country_list, ga.result)

ga.plot_result()
