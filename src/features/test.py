from pathlib import Path
import os
from build_features import Features

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    #print(BASE_DIR)
    STATIC_ROOT = os.path.join(BASE_DIR, r'data')
    #print(STATIC_ROOT)
    f = Features(STATIC_ROOT)
    ds = f.build_features()
    print(ds.shape)

if __name__ == "__main__":
    main()