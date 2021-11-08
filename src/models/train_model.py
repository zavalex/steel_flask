from pathlib import Path
import os
from features.build_features import Features
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import joblib

def dataset_split(path):
    features = Features(path)
    dataset = features.build_features()

    target = dataset['last_measure']
    features = dataset.drop(['key','measure_count','last_measure'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=127)
    return X_train, X_test, y_train, y_test

def train():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODEL_DIR = os.path.join(BASE_DIR, 'models\model.pkl')
    DATA_DIR = os.path.join(BASE_DIR, r'data')

    X_train, X_test, y_train, y_test = dataset_split(DATA_DIR)
    model = CatBoostRegressor(random_state=127, iterations=300, learning_rate=0.08, max_depth=5,num_leaves=31, verbose=0)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    #print(X_test[0])
    #print(y_test[0])
    #print(predict[0])
    print('Test MAE:', mean_absolute_error(y_test, predict))
    print('Test R2:', r2_score(y_test, predict))
    joblib.dump(model,MODEL_DIR)

#if __name__ == "__main__":
#    main()