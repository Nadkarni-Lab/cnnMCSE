"""Generate minimum-convergence sample estimates with convolutional neural networks on simple test cases. 
"""
import argparse
from cnnMCSE.predict import predict_loop

def main(**config_kwargs):
    predict_loop(**config_kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration arguments for convergence sample size estimtation.')
    
    parser.add_argument("--datasets", required=True, type=str,
                        help="Name of datasets.")
    
    parser.add_argument("--models", required=True, type=str,
                        help="Name of models.")
    
    parser.add_argument("--root_dir", required=True, type=str,
                        help="Path to root directory.")
    
    parser.add_argument("--initial_weights_dir", required=True, type=str,
                        help="Path to initial weights directory.")
    
    parser.add_argument("--out_data_path", required=True, type=str,
                        help="Path to output data.")
    
    parser.add_argument("--out_metadata_path", required=True, type=str,
                        help="Path to output metadata with MCSE estimates.")

    parser.add_argument("--zoo_models", required=False, type=str, default=None,
                        help="Name of pretrained model to use.")
    
    parser.add_argument("--n_bootstraps", required=True, type=int,
                        help="Number of bootstraps to conduct.")
    
    parser.add_argument("--batch_size", required=False, type=int, default=4,
                        help="Number of batches.")
    
    parser.add_argument("--n_epochs", required=False, type=int, default=1,
                        help="Number of epochs.")
    
    parser.add_argument("--n_workers", required=False, type=int, default=2,
                        help="Number of workers.")
    
    parser.add_argument("--max_sample_size", required=False, type=int, default=1000,
                        help="Maximum estimated sample size.")
    
    parser.add_argument("--log_scale", required=False, type=int, default=2,
                        help="Log scale to test every sample size.")
    
    parser.add_argument("--min_sample_size", required=False, type=int, default=16,
                        help="Minimum estimated sample size.")
    
    parser.add_argument("--absolute_scale", required=False, type=bool, default=False,
                        help="Run on an absolute rather than log scale.")
    
    parser.add_argument("--num_workers", required=False, type=int, default=4,
                        help="Run on an absolute rather than log scale.")
    
    parser.add_argument("--shuffle", action='store_true', 
                        help="Shuffle the dataset.")

    config_kwargs = parser.parse_args()
    main(**vars(config_kwargs))