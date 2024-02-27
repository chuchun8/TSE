import preprocessor as p, re, wordninja, csv, pandas as pd, os, pdb

def load_data(filename, args):

    filename    = [filename]
    concat_text = pd.DataFrame()
    raw_text    = pd.read_csv(filename[0],usecols=[0], encoding='ISO-8859-1')
    raw_target  = pd.read_csv(filename[0],usecols=[1], encoding='ISO-8859-1')

    labels = {
        'PStance'     : ['Joe Biden', 'Bernie Sanders', 'Donald Trump'],
        'AM'          : ['abortion', 'cloning', 'death penalty', 'gun control', 'marijuana legalization', 'minimum wage', 'nuclear energy', 'school uniforms'],
        'SemEval2016' : ['Atheism', 'Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion'],
        'Covid19'     : ['face masks', 'fauci', 'stay at home orders', 'school closures'],
        'Stance_Merge_Unrelated': ['Atheism', 'Feminist Movement', 'Hillary Clinton', 'abortion', 'face masks', 'fauci', 'stay at home orders', 'school closures', 'cloning', 'death penalty', 'gun control', 'marijuana legalization', 'minimum wage', 'nuclear energy', 'school uniforms', 'Donald Trump', 'Joe Biden', 'Bernie Sanders', 'Unrelated']
        }

    if filename[0].find('_train_') != -1 or filename[0].find('_val_') != -1: # train, val
        target = pd.DataFrame.replace(raw_target, labels[args.dataset], list(range(len(labels[args.dataset]))))
    else:   # test
        target = pd.DataFrame.replace(raw_target, labels[args.dataset], list(range(len(labels[args.dataset]))))
    
    stance              = pd.read_csv(filename[0],usecols=[2], encoding='ISO-8859-1')
    concat_text         = pd.concat([raw_text, target, stance], axis=1)
    concat_text.columns = ['Tweet', 'Target', 'Stance']

    return(concat_text)

def data_clean(text, args, norm_dict):
    
    # remove target strings from the tweets
    replace_strings = {
            'PStance'      : ['Joe', 'Biden', 'Bernie', 'Sanders', 'Donald', 'Trump'],
            'AM'           : ['abortion', 'cloning', 'death', 'penalty', 'gun', 'control', 'marijuana', 'legalization', 'minimum', 'wage', 'nuclear', 'energy', 'school', 'uniforms'],
            'SemEval2016'  : ['Atheism', 'Feminist', 'Movement', 'Hillary',  'Clinton', 'Legalization', 'Abortion'],
            'Covid19'      : ['face', 'masks', 'fauci', 'stay', 'home', 'orders', 'school', 'closures'],
            'Stance_Merge_Unrelated': ['Joe', 'Biden', 'Bernie', 'Sanders', 'Donald', 'Trump', 'abortion', 'cloning', 'death', 'penalty', 'gun', 'control', 'marijuana', 'legalization', 'minimum', 'wage',  'nuclear', 'energy', 'school', 'uniforms', 'Atheism', 'Feminist', 'Movement', 'Hillary', 'Clinton', 'face', 'masks', 'fauci', 'stay', 'home', 'school', 'closures', 'orders']
            }
    
    for x in replace_strings[args.dataset]:
        text = text.lower().replace(x.lower(), '')

    p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.RESERVED)
    
    clean_data = p.clean(text)  # using lib to clean URL, emoji...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+",clean_data)
    clean_data = [[x.lower()] for x in clean_data]
    
    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i][0] = norm_dict[clean_data[i][0]]
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0]) # split compound hashtags
    
    clean_data = [j for i in clean_data for j in i]
    return clean_data

def clean_all(filename, args, norm_dict):
    
    concat_text = load_data(filename, args)
    raw_data    = concat_text['Tweet'].values.tolist() 
    x_target    = concat_text['Target'].values.tolist()
    stance      = concat_text['Stance'].values.tolist()
    clean_data  = [None for _ in range(len(raw_data))]
    
    file = 'output/{}/input.txt'.format(args.dataset)
    if os.path.exists(file):    os.remove(file)
    with open(file, 'a') as f:
        for i in range(len(raw_data)):
            clean_data[i] = data_clean(raw_data[i], args, norm_dict)
            f.write('%s\n' % ' '.join(clean_data[i]))
    
    return clean_data, x_target, stance