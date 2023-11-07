import os
import argparse
import pandas as pd

def main(input_dir, output_directory):
    files = os.listdir(input_dir)

    df_list = list()
    for index, file in enumerate(files):
        df = pd.read_csv(os.path.join(input_dir, file), sep="\t")
        print(index)
        df['bootstrap'] = index
        df_list.append(df)
    
    output_df = pd.concat(df_list)
    output_file = os.path.join(output_directory,  'data' + '.tsv')
    output_df.to_csv(output_file, sep="\t", index=False)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge job for figure 3a.')

    parser.add_argument("--input_dir", required=True, type=str,
                        help="Directory with all tsvs.")
    
    parser.add_argument("--output_directory", required=True, type=str,
                        help="File path with output.")
    config_kwargs = parser.parse_args()

    main(**vars(config_kwargs))