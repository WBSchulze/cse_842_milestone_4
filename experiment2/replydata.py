#@title Define IO utility functions.

import os
import json
import random
import re
import warnings
import pandas as pd

FILTER_INCOMPLETE_REPLIES =     True
FILTER_INCOHERENT_REPLIES =     True
FILTER_DONTKNOW_REPLIES =       True
APPLY_REGEX =                   True
BALANCE_CLASSES =               False

BATCH_SIZE =                    32
NUM_EPOCHS =                    4
LEARNING_RATE =                 5e-5

ROOT =                          'c:\\Users\\William\\Documents\\CSE_842\\'
STORAGE =                       ROOT + 'Project\\'
RAW_DATA_STORAGE =              STORAGE + 'raw_data\\'
RESULTS_STORAGE =               STORAGE + 'results\\'
PROCESSED_DATA_STORAGE =        RESULTS_STORAGE + 'processed_data\\'
DATA_PROCESSING_INFO_STORAGE =  PROCESSED_DATA_STORAGE + 'info\\'
MODEL_STORAGE =                 RESULTS_STORAGE + 'models\\'
FEATURE_IMPORTANCES_STORAGE =   RESULTS_STORAGE + 'feature_importances\\'
MODEL_PERFORMANCES_TITLE =      'model_performances'
FEATURE_IMPORTANCES_CUTOFF =    20

REFUSAL_FLAGS = [
    'Redirected',
    'Rejected',
    'Disclaimed',
    'Contradicted',
    'Counseling',
]

COMPLIANCE_FLAGS = [
    'Complied',
    'Continuing',
]

DATASETS_METADATA = {
    'OpenAI Moderation':    {'source_filename':         'mod_classified.json',
                             'processed_filename':      'mod_classified_processed.json'},
    'New York Post':        {'source_filename':         'nyp_dataset_classified2.jsonl',
                             'processed_filename':      'nyp_dataset_classified2_processed.json'},
    'Political Figures':    {'source_filename':         'political_dataset_classified2.jsonl',
                             'processed_filename':      'political_dataset_classified2_processed.json'},
    'Quora':                {'source_filename':         'quora_classified_all.json',
                             'processed_filename':      'quora_classified_all_processed.json'},
    'Quora (unlabeled)':    {'source_filename':         'quora_large.jsonl',
                             'processed_filename':      'quora_large_processed.json'},
    '4chan':                {'source_filename':         '4chan_classified.ndjson',
                             'processed_filename':      '4chan_classified_processed.json'},
    'All hand-labeled':     {'processed_filename':      'handlabelled.json'},
    'All':                  {'processed_filename':      'all.json'},
}

def load_dialogues( dataset ):
    source_filepath = RAW_DATA_STORAGE + DATASETS_METADATA[dataset]['source_filename']
    diaJsons = read_json_as_list(source_filepath)
    diaDicts, original_size, processed_size, cleaned_prompt_percentage, cleaned_reply_percentage = process_dialogues(diaJsons)

    dialogues = [ { 'x' : f"<prm> {dialogue['prompt']} <rpl> {dialogue['reply']}",
                    'y' : int( dialogue['refusal'] ) }
                        for dialogue in diaDicts ]
    return dialogues

def read_json_as_list(filepath):
    with open(filepath, 'r') as json_file:
        data = [json.loads(json_str) for json_str in json_file]
        if isinstance(data[0], list):
            data = data[0]
        return data

def save_dataset_processing_dataframe_to_csv(dataset_processing_info_dataframe, filename):
    dataset_processing_info_dataframe.to_csv(f'{DATA_PROCESSING_INFO_STORAGE}{filename}', index=False)

def load_dataset_processing_dataframe_from_csv(filename):
    return pd.read_csv(f'{DATA_PROCESSING_INFO_STORAGE}{filename}')

def write_processed_dataset(dataset, dialogues):
    filename = DATASETS_METADATA[dataset]['processed_filename']
    with open(f'{PROCESSED_DATA_STORAGE}{filename}', 'w') as outfile:
        for d in dialogues:
            json.dump(d, outfile)
            outfile.write('\n')

def read_processed_dataset(dataset):
    filename = DATASETS_METADATA[dataset]['processed_filename']
    dialogues = []
    with open(f'{PROCESSED_DATA_STORAGE}{filename}', 'r') as infile:
        for line in infile:
            dialogue = json.loads(line)
            dialogues.append(dialogue)
    return dialogues

def save_feature_importances_dataframe_to_csv(feature_importances_dataframe, dataset, text_type):
    feature_importances_dataframe.to_csv(f'{FEATURE_IMPORTANCES_STORAGE}{dataset}_feature_importances_{text_type}.csv', index=False)

def load_feature_importances_dataframe_from_csv(dataset, text_type):
    return pd.read_csv(f'{FEATURE_IMPORTANCES_STORAGE}{dataset}_feature_importances_{text_type}.csv')

def combine_csv_files(csv_filenames, combined_csv_filename):
    # Initialize an empty DataFrame for combined data
    combined_df = pd.DataFrame()
    
    # Loop through the list of CSV file names and read them into DataFrames
    for csv_filename in csv_filenames:
        if os.path.exists(csv_filename):
            try:
                csv_df = pd.read_csv(csv_filename)
                combined_df = pd.concat([combined_df, csv_df], ignore_index=True)
            except FileNotFoundError:
                print(f"The file '{csv_filename}' does not exist.")
        else:
            print(f"The file '{csv_filename}' does not exist.")
    
    # Write the combined DataFrame to a new CSV file
    combined_df.to_csv(combined_csv_filename, index=False)

def wipe_csv_contents(filename):
    """Wipe the contents of a CSV file."""
    try:
        with open(filename, 'w') as file:
            file.write('')
        print(f"The contents of '{filename}' have been wiped.")
    except FileNotFoundError:
        print(f"The file '{filename}' does not exist.")

#@title Define data-processing category-based-filtering, balancing, and regex functions.

# FILTERING
def filter_incomplete_replies(dialogues): return [dialogue for dialogue in dialogues if dialogue['response']['choices'][0]['finish_reason'] == 'stop'] if not FILTER_INCOMPLETE_REPLIES else dialogues
def filter_incoherent_replies(dialogues): return [dialogue for dialogue in dialogues if not dialogue.get('Incoherent', False)] if not FILTER_INCOHERENT_REPLIES else dialogues
def filter_dontknow_replies(dialogues): return [dialogue for dialogue in dialogues if not dialogue.get('DontKnow', False)] if not FILTER_DONTKNOW_REPLIES  else dialogues

# BALANCING
def balance_classes(dialogues):
    if not BALANCE_CLASSES:
        return dialogues

    # Filter dialogues that do not have the 'refusal' key
    no_refusal_dialogues = [dialogue for dialogue in dialogues if dialogue.get('refusal') is None]
    if no_refusal_dialogues == []:
        return dialogues

    refusals = [dialogue for dialogue in dialogues if dialogue['refusal']]
    compliances = [dialogue for dialogue in dialogues if not dialogue['refusal']]
    
    total_samples = len(dialogues)
    refusals_percentage = (len(refusals) / total_samples) * 100
    
    majority_class_size = max(len(refusals), len(compliances))
    minority_class_size = min(len(refusals), len(compliances))
    
    if len(refusals) < len(compliances):
        minority_class = refusals
        majority_class = compliances
    else:
        minority_class = compliances
        majority_class = refusals
    
    upsampled_minority_class = minority_class * (majority_class_size // minority_class_size) + random.sample(minority_class, majority_class_size % minority_class_size)
    
    balanced_classes = upsampled_minority_class + majority_class
    random.shuffle(balanced_classes)
    return balanced_classes

def clean_text(text, text_type):
    # Define patterns and replacements
    replacements = [
        (r'&gt;&gt;\d+|&gt;&gt;|&gt;|&quot;', ' '),
        (r'&#039;', '\''),
        (r'&#8221;', ''),
        (r'&#8220;', '')
    ]
    if text_type == 'prompt':
        replacements.append((r'<[^>]+>', ''))
    # Apply replacements
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    # Remove extra spaces, but preserve new lines
    if text_type == 'prompt':
        return re.sub(r'[^\S\n]+', ' ', text)
    else:
        return text

def apply_regex(dialogues):
    prompts_changed = 0
    replies_changed = 0
    for dialogue in dialogues:
        original_prompt = dialogue['prompt']
        original_reply = dialogue['reply']
        cleaned_prompt = clean_text(original_prompt, 'prompt')
        cleaned_reply = clean_text(original_reply, 'reply')

        if original_prompt != cleaned_prompt:
            prompts_changed += 1
        if original_reply != cleaned_reply:
            replies_changed += 1

        dialogue['prompt'] = cleaned_prompt
        dialogue['reply'] = cleaned_reply

    return 100 * prompts_changed / len(dialogues), 100 * replies_changed / len(dialogues)


#@title Define data-processing execution.

def process_dialogues(dialogues):
    for dialogue in dialogues:
        dialogue['reply'] = dialogue['response']['choices'][0]['message']['content']  # for simpler referencing
    
    original_dataset_size = len(dialogues)

    cleaned_prompt_percentage, cleaned_reply_percentage = 0, 0
    if APPLY_REGEX:
        cleaned_prompt_percentage, cleaned_reply_percentage = apply_regex(dialogues)

    dialogues = filter_incomplete_replies(dialogues)

    for dialogue in dialogues:
        dialogue['refusal'] = any(dialogue.get(flag) == 1 for flag in REFUSAL_FLAGS)

    dialogues = balance_classes(dialogues)
    return dialogues, original_dataset_size, len(dialogues), cleaned_prompt_percentage, cleaned_reply_percentage

def preprocess_labeled_data():
    filename = 'labeled_data_preprocessing.csv'
    warnings.filterwarnings('ignore')
    dataset_processing_info_dataframe = pd.DataFrame()

    for dataset in ['OpenAI Moderation', 'New York Post', 'Political Figures', 'Quora', '4chan']:
        source_filepath = RAW_DATA_STORAGE + DATASETS_METADATA[dataset]['source_filename']
        dialogues = read_json_as_list(source_filepath)
        dialogues, original_size, processed_size, cleaned_prompt_percentage, cleaned_reply_percentage = process_dialogues(dialogues)

        dataset_processing_info_dataframe = dataset_processing_info_dataframe.append({
            'Dataset': dataset,
            'Original_Size': original_size,
            'Processed_Size': processed_size,
            'Cleaned_Prompt_Percentage': cleaned_prompt_percentage,
            'Cleaned_Reply_Percentage': cleaned_reply_percentage,
        }, ignore_index=True)

        write_processed_dataset(dataset, dialogues)

    ############################################################################
    # combine all the data into the 'All hand-labeled' class
    all_dialogues = []
    for dataset in ['OpenAI Moderation', 'New York Post', 'Political Figures', 'Quora', '4chan']:
        all_dialogues += read_processed_dataset(dataset)
    all_dialogues, original_size, processed_size, cleaned_prompt_percentage, cleaned_reply_percentage = process_dialogues(all_dialogues)

    dataset_processing_info_dataframe = dataset_processing_info_dataframe.append({
        'Dataset': 'All hand-labeled',
        'Original_Size': original_size,
        'Processed_Size': processed_size,
        'Cleaned_Prompt_Percentage': cleaned_prompt_percentage,
        'Cleaned_Reply_Percentage': cleaned_reply_percentage,
    }, ignore_index=True)
    
    write_processed_dataset('All hand-labeled', all_dialogues)
    ############################################################################

    save_dataset_processing_dataframe_to_csv(dataset_processing_info_dataframe, filename)
    # display_full_dataset_processing_info()