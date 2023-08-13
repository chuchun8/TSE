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
    tar_label = pd.read_csv(filename,usecols=[3], encoding='ISO-8859-1')
    label = pd.DataFrame.replace(raw_label,['Dummy Stance','FAVOR','NONE','AGAINST'], [3,2,1,0])
    concat_text = pd.concat([raw_text, label, raw_target], axis=1)
    concat_text_aux = pd.concat([raw_text, tar_label, label], axis=1)
    concat_text_aux = concat_text_aux[concat_text_aux['Stance'] != 1] # remove 'NONE' label for aux task
    
    return concat_text, concat_text_aux


# Data cleaning
def data_clean(strings, norm_dict, task):
    
    # Target lists for aux task without wtwt dataset
    replace_strings = ['Joe', 'Biden', 'Bernie', 'Sanders', 'Donald', 'Trump', 'abortion', 'cloning', \
                       'death', 'penalty', 'gun', 'control', 'marijuana', 'legalization', 'minimum', 'wage', \
                       'nuclear', 'energy', 'school', 'uniforms', 'Atheism', 'Feminist', 'Movement', \
                       'Hillary', 'Clinton', 'face', 'masks', 'fauci', 'stay','home', 'school', 'closures']
            
    if task == "aux":
        for x in replace_strings:
            strings = strings.lower().replace(x.lower(), '') # remove the target mentions from the text for aux task
    
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)
    clean_data = p.clean(strings) # using lib to clean URL,hashtags...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.sub(r"#semst", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
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
    
    concat_text, concat_text_aux = load_data(filename)
    
    # Main task
    raw_data = concat_text['Tweet'].values.tolist()
    label = concat_text['Stance'].values.tolist()
    x_target = concat_text['Target'].values.tolist()
    clean_data = [None for _ in range(len(raw_data))]
    
    print("data size in main task: ", len(raw_data))
    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i], norm_dict, 'main')
        x_target[i] = data_clean(x_target[i], norm_dict, 'main')
    
    # Auxiliary target prediction task
    raw_data_aux = concat_text_aux['Tweet'].values.tolist()
    label_aux = concat_text_aux['ID'].values.tolist()
    clean_data_aux = [None for _ in range(len(raw_data_aux))]
    
    print("data size in auxiliary task: ", len(raw_data_aux))
    for i in range(len(raw_data_aux)):
        clean_data_aux[i] = data_clean(raw_data_aux[i], norm_dict, 'aux')
    
    return clean_data, label, x_target, clean_data_aux, label_aux
    