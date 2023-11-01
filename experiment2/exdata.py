import os
from io import open
import torch
import random

class Dictionary(object):
    def __init__(self):
        self.word2idx = { 'n/a' : 0 }
        self.idx2word = [ 'n/a' ]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, dialogues, seqLength = 512):
        self.dictionary = Dictionary()
        self.seqLength = seqLength

        testNum = len( dialogues ) // 8 # integer division
        trainDialogues = dialogues[:-2*testNum]
        self.iComplied = [ iDialogue for iDialogue, dialogue in enumerate(trainDialogues) if dialogue['y'] == 0]
        self.iRefused = [ iDialogue for iDialogue, dialogue in enumerate(trainDialogues) if dialogue['y'] == 1]
        self.trainX, self.trainY = self.tokenize( trainDialogues)

        self.validX, self.validY = self.tokenize(dialogues[-2*testNum:-testNum])
        self.testX, self.testY = self.tokenize(dialogues[-testNum:])
        
    def chooseTrain( self ):
        if random.random() > 0.5:
            iChoice = random.choice( self.iComplied )
        else:
            iChoice = random.choice( self.iRefused )
        return iChoice

    def tokenize(self, dialogues):
        """Tokenizes a text file."""
        # Add words to the dictionary
        for dialogue in dialogues:
            words = dialogue['x'].split() + ['<eos>']
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        idss = []
        for dialogue in dialogues:
            words = dialogue['x'].split() + ['<eos>']
            ids = [0] * self.seqLength
            for iWord, word in enumerate( words ):
                ids[iWord] = self.dictionary.word2idx[word]
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.vstack(idss)

        ys = torch.tensor( [ float(dialogue['y']) for dialogue in dialogues ] )

        return ids, ys
