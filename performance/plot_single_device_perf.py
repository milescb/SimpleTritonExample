
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# ploting style
import mplhep as hep
plt.style.use(hep.style.ATLAS)
plt.rcParams['legend.loc'] = 'upper left'
figsize = (7, 8)
colors = plt.get_cmap('tab10')

def clean_pandas_df(df):
    return df.sort_values(by='Concurrency', ascending=True)

def instance_number(filename):
    match = re.search(r'(cpu|gpu)_(\d+)instance_sync\.csv', filename)
    if match:
        return int(match.group(2))
    else:
        return None

def process_csv_dir(directory, one_gpu=True):
    gpu_data_instances = {}
    cpu_data_instances = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.csv'):
                if one_gpu and '_1gpus' not in root:
                    continue
                file_path = os.path.join(root, filename)
                if 'gpu' in filename:
                    gpu_data = pd.read_csv(file_path)
                    gpu_data = clean_pandas_df(gpu_data)
                    gpu_data_instances[instance_number(filename)] = gpu_data
                elif 'cpu' in filename:
                    cpu_data = pd.read_csv(file_path)
                    cpu_data = clean_pandas_df(cpu_data)
                    cpu_data_instances[instance_number(filename)] = cpu_data

    if not gpu_data_instances and not cpu_data_instances:
        raise ValueError(f"No csv files found in {directory}")

    return cpu_data_instances, gpu_data_instances

def plot_var_vs_instance(data_dict, 
                         variable='Inferences/Second', 
                         ylabel='Throughput (infer/sec)',
                         title="AMD Radeon Pro W7700",
                         outdir="data/",
                         save_name='instances_vs_throughput_gpu.pdf',
                         ratio=True,
                         save=True):
    
    instances = sorted(data_dict.keys())
    concurrencies = data_dict[1]['Concurrency'].values
    concurrencies = concurrencies[:10]
    
    con_vals = []
    for con in concurrencies:
        vals = []
        for i in instances:
            val = data_dict[i][data_dict[i]['Concurrency'] == con][variable].values
            if len(val) == 0:
                print(f'No data for {i} instances and {con} \
                    concurrent requests, setting to 0')
                vals.append(0.0)
            else:
                vals.append(val[0])
        con_vals.append(vals)
    
    if ratio:
        
        ratio_vals = []
        for i in range(1, len(concurrencies)):
            ratio_vals.append([x/y for x, y in zip(con_vals[i], con_vals[0])])
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       sharex=True, 
                                       gridspec_kw={'height_ratios': [4, 1]})
        
        for i, con in enumerate(concurrencies):
            ax1.plot(instances, con_vals[i], 'o-', 
                     color=colors(i), label=f'{con} requests')
            
        for i, con in enumerate(concurrencies[1:]):
            ax2.plot(instances, ratio_vals[i], 'o-', 
                     color=colors(i+1), label=f'{con} requests')
            
        ax1.set_ylabel(ylabel, loc='top')
        ax1.set_title(title, loc='left', fontsize=12)
        ax1.legend()
        
        ax2.set_xlabel('Number of Triton Model Instances', loc='right')
        ax2.set_ylabel('Ratio')
        
        plt.subplots_adjust(hspace=0.07) 
        
        if save:
            plt.savefig(f'{outdir}/{save_name}', bbox_inches='tight')
        
    else:
        plt.figure(figsize=(5, 5))
        for con in concurrencies:
            plt.plot(instances, con_vals[con], 'o-', label=f'{con} \
                concurrent requests')
        plt.xlabel('Number of Triton Model Instances', loc='right')
        plt.ylabel(ylabel, loc='top')
        plt.legend()
        
        if save:
            plt.savefig(f'{outdir}/{save_name}', bbox_inches='tight')
            
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputdir", default="data/nominal_alpaka/")
    parser.add_argument("-o", "--outputdir", default=None)
    parser.add_argument("-t", "--title", default="AMD Radeon Pro W7700")
    args = parser.parse_args()

    INPUT_DIR = args.inputdir
    OUTPUT_DIR = args.outputdir if args.outputdir is not None else INPUT_DIR

    cpu_data_instances, gpu_data_instances = process_csv_dir(INPUT_DIR)

    plot_var_vs_instance(gpu_data_instances, outdir=OUTPUT_DIR, title=args.title)
    
if __name__=="__main__":
    main()
    
    