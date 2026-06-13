from src.prepare_real_data import main as prepare_main
from src.train_model import main as train_main
from src.predict import main as predict_main

if __name__ == "__main__":
    prepare_main()
    train_main()
    predict_main()
