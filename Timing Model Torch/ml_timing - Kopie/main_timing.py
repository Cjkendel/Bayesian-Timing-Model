from models import LatentTimingCluster

if __name__ == '__main__':
    training = LatentTimingCluster(training_dataset='data.csv')
    loss, clusters = training.fit(500)
    print(f'The final loss was : {loss[-1]} \n Final updated cluster paramaters are:     {clusters}')
