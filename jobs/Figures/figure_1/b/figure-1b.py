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
    
    parser.add_argument("--metric_type", required=False, type=str, default='AUC',
                        help="Metric Type to evaluate on.")

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
    
    parser.add_argument("--absolute_scale", required=False, type=int, default=None,
                        help="Run on an absolute rather than log scale.")
    
    parser.add_argument("--shuffle", action='store_true', 
                        help="Shuffle the dataset.")
    
    parser.add_argument("--frequency", action='store_true', 
                        help="Calculate dataset frequencies.")
    
    parser.add_argument("--stratified", action='store_true', 
                        help="Calculate stratified losses for minimum convergence sample estimation.")


    config_kwargs = parser.parse_args()
    main(**vars(config_kwargs))


def initial_command():
    return """python cnn-frequency-sauc.py --datasets MNIST --metric_type sAUC --models A3,FCN --root_dir /mnt/c/Users/faris/Documents/sinai/Research/Nadkarni/cnnMCSE/data  --n_bootstraps 1 --shuffle --batch_size 1 --n_workers 0 --max_sample_size 500 --frequency  --initial_weights_dir /mnt/c/Users/faris/Documents/sinai/Research/Nadkarni/cnnMCSE/jobs/2_cnn-smcse-experiments/2.1_cnn-smcse-sAUC/weights --out_data_path /mnt/c/Users/faris/Documents/sinai/Research/Nadkarni/cnnMCSE/jobs/2_cnn-smcse-experiments/2.1_cnn-smcse-sAUC/output/data.tsv --out_metadata_path /mnt/c/Users/faris/Documents/sinai/Research/Nadkarni/cnnMCSE/jobs/2_cnn-smcse-experiments/2.1_cnn-smcse-sAUC/output/fcn_metadata.tsv"""