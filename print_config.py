import pickle

res = pickle.load(open( "log/exp_2019-08-17T23:57:55.168006/bohb_result.pkl", "rb" ))
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
print(id2config[incumbent]['config'])