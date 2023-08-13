import preprocessor as p 
import re
import wordninja
import csv
import pandas as pd


# Data loading
def load_data(filename):

    concat_text = pd.DataFrame()
    raw_text = pd.read_csv(filename,usecols=[0], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename,usecols=[1], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename,usecols=[2], encoding='ISO-8859-1')
    mapped_tar = pd.read_csv(filename,usecols=[3], encoding='ISO-8859-1')
    label = pd.DataFrame.replace(raw_label,['Dummy Stance','FAVOR','NONE','AGAINST'], [3,2,1,0])
    concat_text = pd.concat([raw_text, label, raw_target, mapped_tar], axis=1)
    
    return concat_text

# Data cleaning
def data_clean(strings, norm_dict):
    
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)
    clean_data = p.clean(strings) # using lib to clean URL,hashtags...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.sub(r"#semst", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+",clean_data)
    clean_data = [[x.lower()] for x in clean_data]
    
    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0]) # separate hashtags
    clean_data = [j for i in clean_data for j in i]

    return clean_data

# Clean all data
def clean_all(filename, norm_dict):
    
    concat_text = load_data(filename)
    
    raw_data = concat_text['Tweet'].values.tolist()
    label = concat_text['GT Stance'].values.tolist()
    x_target = concat_text['GT Target'].values.tolist()
    x_mapped_tar = concat_text['Mapped Target'].values.tolist()
    clean_data = [None for _ in range(len(raw_data))]
    
    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i], norm_dict)
        x_target[i] = data_clean(x_target[i], norm_dict)
        x_mapped_tar[i] = data_clean(x_mapped_tar[i], norm_dict)
    
    return clean_data, label, x_target, x_mapped_tar