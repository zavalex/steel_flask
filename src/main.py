from os import terminal_size
from models.train_model import train
from models.predict_model import predict_with_trained_model
import numpy as np

def main():
    data = [1571,861,5,0.31]
    data = np.array(data)
    pred = predict_with_trained_model(data)
    print(pred)

if __name__ == "__main__":
    main()