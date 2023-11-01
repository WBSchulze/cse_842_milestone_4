import replydata
import warnings
import pandas as pd
import exdata


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

filename = 'labeled_data_preprocessing.csv'
warnings.filterwarnings('ignore')

# ================================
# Load the dialogues.
# ================================

dataset = 'Quora'
source_filepath = RAW_DATA_STORAGE + replydata.DATASETS_METADATA[dataset]['source_filename']
diaJsons = replydata.read_json_as_list(source_filepath)
diaDicts, original_size, processed_size, cleaned_prompt_percentage, cleaned_reply_percentage = replydata.process_dialogues(diaJsons)

dialogues = [ { 'x' : f"<prm> {dialogue['prompt']} <rpl> {dialogue['reply']}",
                'y' : int( dialogue['refusal'] ) }
                    for dialogue in diaDicts ]

corpus = exdata.Corpus( dialogues )




