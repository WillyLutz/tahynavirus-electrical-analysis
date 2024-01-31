import datetime
import os

from fiiireflyyy.files import get_all_files
import fiiireflyyy.learn as fl
import fiiireflyyy.process as fp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from scripts.pipelines import confusion, pca, feature_importance

save = "/media/wlutz/TOSHIBA EXT/Electrical activity analysis/TAHINAVIRUS/RESULTS/Review + donors D+E"
merge = "datasets/merge ni tahv rg27 t0 t30 t48 all donors separated.csv"

dataset_path = "/media/wlutz/TOSHIBA EXT/Electrical activity analysis/TAHINAVIRUS/DATASETS/"

def main():
    df = pd.read_csv(dataset_path+"NI T48 All donors slices separated.csv")
    df = df.reset_index(drop=True)
    # slices = ["Slice 1", "Slice 2", "Slice 3"]
    # slices = ["Slice 4", "Slice 5", "Slice 6"]
    # slices = ["Slice 7", "Slice 8", "Slice 9"]
    # slices = ["Slice 10", "Slice 11", "Slice 12"]
    
    df = df[df["label"]!='Slice 7']

    indices_to_smooth = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 51, 57, 63, 69, 75, 81, 87, 93,
                         100, 106, 112, 118, 124, 130, 136, 142, 148, 154, 160, 166]
    for i in indices_to_smooth:
        df[str(i)] = df[[str(i-1), str(i+1)]].mean(axis=1)

    df['label'] = df['label'].replace("Slice 1", "Donor A")
    df['label'] = df['label'].replace("Slice 2", "Donor A")
    df['label'] = df['label'].replace("Slice 3", "Donor A")
    df['label'] = df['label'].replace("Slice 4", "Donor B")
    df['label'] = df['label'].replace("Slice 5", "Donor B")
    df['label'] = df['label'].replace("Slice 6", "Donor B")
    df['label'] = df['label'].replace("Slice 7", "Donor B'")
    df['label'] = df['label'].replace("Slice 8", "Donor B'")
    df['label'] = df['label'].replace("Slice 9", "Donor B'")
    df['label'] = df['label'].replace("Slice 10", "Donor C")
    df['label'] = df['label'].replace("Slice 11", "Donor C")
    df['label'] = df['label'].replace("Slice 12", "Donor C")

    
    train = [list(set(list(df["label"])))]
    
    percentiles = 0.1

    # random split train test for train labels
    # discarding outliers
   
    
    pca, pcdf, ratio = fl.fit_pca(df, 2)
    

    # random split train test for test labels
    df_test_labels = pd.DataFrame()
    test_pcdf = pd.DataFrame()
    
    
    fl.plot_pca(pd.concat([pcdf, test_pcdf], ignore_index=True), n_components=2,
                show=True,
                metrics=True,
                savedir='',
                title='',
                ratios=[round(x*100, 2) for x in ratio],
                dpi=300)
    
    
    # for s in range(len(slices)):
    #     sliceA = df[df["label"] == slices[s]]
    #     sliceA_mean = sliceA.mean(axis=0)
    #     sliceA_std = sliceA.std(axis=0)
    #     plt.plot(sliceA_mean,label=slices[s])
    #     plt.fill_between(x=[x for x in range(len(sliceA_mean))], y1=sliceA_mean.sub(sliceA_std), y2=sliceA_mean.add(sliceA_std), alpha=0.5)
    # for donor in list(set(list(df["label"]))):
    #     sliceA = df[df["label"] == donor]
    #     sliceA_mean = sliceA.mean(axis=0)
    #     sliceA_std = sliceA.std(axis=0)
    #     plt.plot(sliceA_mean,label=donor)
    #     plt.fill_between(x=[x for x in range(len(sliceA_mean))], y1=sliceA_mean.sub(sliceA_std), y2=sliceA_mean.add(sliceA_std), alpha=0.5)
    # plt.legend()
    # plt.show()
    #
    pcdf.to_csv(os.path.join("/media/wlutz/TOSHIBA EXT/Papers/Tahynavirus Basia", f"Figure 3I.csv"), index=False)
    
    
    
    
    
    
    # for donor in ['A', 'B', 'C', 'D', 'E']:
    #     confusion([f"NI T48 {donor}", f"TAHV T48 {donor}"], [f"TAHV RG27 T48 {donor}"], merge_path=merge,
    #               savepath=save,
    #               title=f'CONFUSION RG27 effect donor {donor} 0-60 features',
    #               show=False)
    #
    #     pca([f"NI T48 {donor}", f"TAHV T48 {donor}"], [f"TAHV RG27 T48 {donor}"], merge_path=merge,
    #               savepath=save,
    #               title=f'PCA RG27 effect donor {donor} 0-60 features', show=False)
    #     feature_importance([f"NI T48 {donor}", f'TAHV T48 {donor}'],
    #                        mode='impurity',
    #                        savepath=save,
    #                        title=f'FEATURE IMPORTANCE donor {donor} impurity 0-60 features',
    #                        show=False)
    #
    #     print(donor, " done")


now = datetime.datetime.now()
print(now)
main()

# todo : update test_clf_by_confusion, train_RFC_from_dataset, apply_pca, fit_pca

print("RUN:", datetime.datetime.now() - now)
