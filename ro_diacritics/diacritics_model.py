import torch
import torch.nn as nn


class Diacritics(nn.Module):
    """
    The pytorch model :class:Diacritics for diacritics restoration
        :param nr_classes:
        :param word_embedding_size:
        :param character_embedding_size:
        :param char_vocabulary_size:
        :param char_padding_index:
        :param character_window:
        :param sentence_window:
        :param characters_lstm_size:
        :param sentence_lstm_size:
        :param dropout:

    """
    def __init__(self, nr_classes=3, word_embedding_size=300,
                 character_embedding_size=20, char_vocabulary_size=771, char_padding_index=0,
                 character_window=13, sentence_window=31,
                 characters_lstm_size=64, sentence_lstm_size=256, dropout=0.2):
        super(Diacritics, self).__init__()
        self.nr_classes = nr_classes
        self.character_embedding_size = character_embedding_size
        self.char_vocabulary_size = char_vocabulary_size
        self.word_embedding_size = word_embedding_size
        self.char_padding_index = char_padding_index
        self.characters_lstm_size = characters_lstm_size
        self.sentence_lstm_size = sentence_lstm_size
        self.character_window = character_window
        self.sentence_window = sentence_window
        self.dropout = dropout

        self.embedding = nn.Embedding(self.char_vocabulary_size, self.character_embedding_size, self.char_padding_index)
        self.character_lstm_layer = nn.LSTM(
            input_size=self.character_embedding_size,
            hidden_size=self.characters_lstm_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.sentence_bi_lstm_layer = nn.LSTM(
            input_size=self.word_embedding_size,
            hidden_size=self.sentence_lstm_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        concat_size = 2*self.characters_lstm_size + 2*self.sentence_lstm_size + self.word_embedding_size

        self.drop = nn.Dropout(p=self.dropout)

        self.dense = nn.Linear(concat_size, self.nr_classes)

    def forward(self, char_input, word_embedding, sentence_embedding):
        """
        :param char_input: shape=(self.window_character * 2 + 1,)
        :param word_embedding: shape=(word_embedding_size,)
        :param sentence_embedding: shape=(self.window_sentence * 2 + 1, self.word_embedding_size,)
        :return:
        """
        char_emb = self.embedding(char_input)
        device = next(self.parameters()).device

        h = torch.zeros((2*self.character_lstm_layer.num_layers, char_emb.size(0), self.character_lstm_layer.hidden_size))
        c = torch.zeros((2*self.character_lstm_layer.num_layers, char_emb.size(0), self.character_lstm_layer.hidden_size))
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        char_hidden, _ = self.character_lstm_layer(char_emb, (h.to(device), c.to(device)))

        h = torch.zeros((2*self.sentence_bi_lstm_layer.num_layers, sentence_embedding.size(0), self.sentence_bi_lstm_layer.hidden_size))
        c = torch.zeros((2*self.sentence_bi_lstm_layer.num_layers, sentence_embedding.size(0), self.sentence_bi_lstm_layer.hidden_size))
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        sentence_hidden, _ = self.sentence_bi_lstm_layer(sentence_embedding, (h.to(device), c.to(device)))

        concatenated = torch.cat((char_hidden[:, -1, :], word_embedding ,sentence_hidden[:, -1, :]), dim=-1)

        concatenated = self.drop(concatenated)
        out = self.dense(concatenated)

        # out = F.softmax(out, dim=-1)

        return out

    def save(self, filename, vocabulary, epoch=None, valid_acc=0.0, valid_f1=0.0):
        to_save = {'epoch': epoch, 'model_state': self.state_dict(),
                   'vocabulary': vocabulary,
                   'hyperparams': {
                       'nr_classes': self.nr_classes,
                       'character_embedding_size': self.character_embedding_size,
                       'char_vocabulary_size': self.char_vocabulary_size,
                       'word_embedding_size': self.word_embedding_size,
                       'char_padding_index': self.char_padding_index,
                       'characters_lstm_size': self.characters_lstm_size,
                       'sentence_lstm_size': self.sentence_lstm_size,
                       'character_window': self.character_window,
                       'sentence_window': self.sentence_window,
                   }, 'valid_acc': valid_acc, "valid_f1": valid_f1}
        torch.save(to_save, filename)

