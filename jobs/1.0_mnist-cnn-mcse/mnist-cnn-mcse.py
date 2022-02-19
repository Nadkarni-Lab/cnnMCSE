"""Generate minimum-convergence sample estimates with convolutional neural networks on simple test cases. 
"""
import argparse
from cnnMCSE.predict import predict_loop

def main(**config_kwargs):
    predict_loop(**config_kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration arguments for convergence sample size estimtation.')
    
    parser.add_argument("--dataset", required=True, type=str,
                        help="Name of dataset.")
    
    parser.add_argument("--models", required=True, type=str,
                        help="Name of models.")
    
    parser.add_argument("--root_dir", required=True, type=str,
                        help="Path to root directory.")

    config_kwargs = parser.parse_args()
    main(**vars(config_kwargs))