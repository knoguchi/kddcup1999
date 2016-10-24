from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

from pprint import pprint
from time import time
from numpy import array

def init_sparkContext(verbose=False):
    
    conf = SparkConf()
    conf.setAppName('ML')
    conf.set('spark.storage.memoryFraction', '0.5')
    
    # Set a Java system property, such as spark.executor.memory.
    SparkContext.setSystemProperty('spark.executor.memory', '2g')

    # create Spark Context
    sc = SparkContext("local[*]", conf=conf)
    
    # print sc environment
    if verbose:
        pprint(sc._conf.getAll())
        print("Parallelism {}".format(sc.defaultParallelism))

    return sc

def parse_interaction(line):
    # parse log line.  Example:
    # 0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.

    line_split = line.split(",")
    # leave_out categorical values [1,2,3,41] = [tcp, http, SF, normal]
    clean_line_split = line_split[0:1]+line_split[4:41]
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0

    # LabeledPoint ( teacher_signal, feature_row )
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

def train(sc, training_data):
    try:
        logit_model = LogisticRegressionModel.load(sc, "logit_model")
        return logit_model
    except:
        pass
    
    logit_model = LogisticRegressionWithLBFGS.train(training_data)
    
    # save the model
    logit_model.save(sc, "logit_model")
    
    return logit_model

def evaluate(sc, logit_model, test_data):
    labels_and_preds = test_data.map(lambda p: (p.label, logit_model.predict(p.features)))
    return labels_and_preds

def load_data_file(sc, data_file):
    data = sc.textFile(data_file)
    print "Data size is {}".format(data.count())
    return data.map(parse_interaction)
    
def main():
    sc = init_sparkContext(verbose=True)

    # training
    training_data = load_data_file(sc, './data/kddcup.data.gz')
    #training_data = load_data_file(sc, './data/small.gz')
    
    t0 = time()
    logit_model = train(sc, training_data)
    tt = time() - t0
    
    print "Classifier trained in {} seconds".format(round(tt,3))

    # evaluating
    test_data = load_data_file(sc, './data/corrected.gz')
    labels_and_preds = evaluate(sc, logit_model, test_data)

    pprint(labels_and_preds.collect())

main()
