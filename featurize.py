
from lda2vec import utils
from lda2vec import prepare_topics, print_top_words_per_topic, topic_coherence
from lda2vec import preprocess
from lda2vec import Corpus
from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood
from lda2vec.utils import move
from model import LDA2Vec
import numpy as np
import logging
import chainer
from chainer import cuda
import chainer.optimizers as O
import time
import shelve
import logging
from collections import defaultdict
from gensim.models.word2vec import KeyedVectors

gpu_id = int(os.getenv('CUDA_GPU', 0))
cuda.get_device(gpu_id).use()
print "Using GPU " + str(gpu_id)


class Lda2VecFeaturizer:
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, clambda=200, n_topics=10, batchsize=4096, power=0.75, words_pretrained=True, temperature=1,
                 max_length=1000, min_count=0, word2vec_path=None):
        """
        Put class description and init procedures here
        
        :param clambda: int, 'Strength' of the dircihlet prior; 200.0 seems to work well 
        :param n_topics: int, Number of topics to fit
        :param batchsize: 
        :param power: int, Power for neg sampling
        :param words_pretrained: 
        :param temperature: 
        :param max_length: 
        :param min_count: 
        :param word2vec_path: 
        """
        
        self.clambda = clambda
        self.n_topics = n_topics
        self.batchsize = batchsize
        self.power = power  #float(os.getenv('power', 0.75))
        # Intialize with pretrained word vectors
        self.words_pretrained = words_pretrained  #bool(int(os.getenv('pretrained', True)))
        self.temp = temperature
        self.max_length = max_length
        self.min_count = min_count
        self.word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    # ------------------------------------------------------------------------------------------------------------------
    def preprocess(self, docs=None):
        """
        
        :param docs: 
        :return: 
        """
        assert (isinstance(docs, list)), ("input list of documents")
        assert (all(isinstance(doc, unicode) for doc in docs)),("expected unicode, got string")
        
        self.corpus = Corpus()
        tokens, self.vocab = preprocess.tokenize(docs, self.max_length, merge=False,n_threads=4)
        
        # Make a ranked list of rare vs frequent words
        self.corpus.update_word_count(tokens)
        self.corpus.finalize()
        # The tokenization uses spaCy indices, and so may have gaps
        # between indices for words that aren't present in our dataset.
        # This builds a new compact index
        compact = self.corpus.to_compact(tokens)
        # Remove extremely rare words
        pruned = self.corpus.filter_count(compact, min_count=0)
        # Convert the compactified arrays into bag of words arrays
        bow = self.corpus.compact_to_bow(pruned)
        # Words tend to have power law frequency, so selectively
        # downsample the most prevalent words
        clean = self.corpus.subsample_frequent(pruned)
        # Now flatten a 2D array of document per row and word position per column to a 1D array of words
        # This will also remove skips and OoV words
        self.doc_ids = np.arange(pruned.shape[0])
        self.flattened, (self.doc_ids,) = self.corpus.compact_to_flat(pruned, self.doc_ids)

        self.vectors, s, f = self.corpus.compact_word_vectors(self.vocab, model = self.word2vec_model)
        # vectors = np.delete(vectors,77743,0)
        self.n_docs = len(docs) #doc_ids.max() + 1
        # Number of unique words in the vocabulary
        self.n_vocab=self.flattened.max()  + 1

        doc_idx, lengths = np.unique(self.doc_ids, return_counts=True)
        self.doc_lengths = np.zeros(self.doc_ids.max() + 1, dtype='int32')
        self.doc_lengths[doc_idx] = lengths
        # Count all token frequencies
        tok_idx, freq = np.unique(self.flattened, return_counts=True)
        self.term_frequency = np.zeros(self.n_vocab, dtype='int32')
        self.term_frequency[tok_idx] = freq

        self.fraction = self.batchsize * 1.0 / self.flattened.shape[0]

        # Get the string representation for every compact key
        self.words = self.corpus.word_list(self.vocab)[:self.n_vocab]

    # ------------------------------------------------------------------------------------------------------------------
    def train(self,docs=None, epochs=200, update_words=False, update_topics=True):
        """
        
        :param docs: 
        :param epochs: 
        :param update_words: 
        :param update_topics: 
        :return: 
        """
        logging.info("preprocessing...")
        self.preprocess(docs)
        logging.info('preprocessed!')
        
        self.train_model = LDA2Vec(n_documents=self.n_docs,\
                        n_document_topics=self.n_topics,\
                        n_units=300,\
                        n_vocab=self.n_vocab,\
                        counts=self.term_frequency,\
                        n_samples=15,\
                        power=self.power,\
                        temperature=self.temp)

        if self.words_pretrained:
            self.train_model.sampler.W.data[:, :] = self.vectors[:self.n_vocab, :]

        self.train_model.to_gpu()

        optimizer = O.Adam()
        optimizer.setup(self.train_model)
        clip = chainer.optimizer.GradientClipping(5.0)
        optimizer.add_hook(clip)
        
        j = 0
        msgs = defaultdict(list)
        
        for epoch in range(epochs):
            print "epoch : ",epoch
            data = prepare_topics(cuda.to_cpu(self.train_model.mixture.weights.W.data).copy(),
                                  cuda.to_cpu(self.train_model.mixture.factors.W.data).copy(),
                                  cuda.to_cpu(self.train_model.sampler.W.data).copy(),
                                  self.words)
            top_words = print_top_words_per_topic(data)
            if j % 100 == 0 and j > 100:
                coherence = topic_coherence(top_words)
                for j in range(self.n_topics):
                    print j, coherence[(j, 'cv')]
                kw = dict(top_words=top_words, coherence=coherence, epoch=epoch)
                #progress[str(epoch)] = pickle.dumps(kw)
            data['doc_lengths'] = self.doc_lengths
            data['term_frequency'] = self.term_frequency
            #np.savez('topics.pyldavis', **data)
            for d, f in utils.chunks(self.batchsize, self.doc_ids, self.flattened):
                t0 = time.time()
                optimizer.zero_grads()
                l = self.train_model.fit_partial(d.copy(), f.copy(), update_words=update_words, update_topics=update_topics)
                prior = self.train_model.prior()
                loss = prior * self.fraction
                loss.backward()
                optimizer.update()
                msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
                       "P:{prior:1.3e} R:{rate:1.3e}")
                prior.to_cpu()
                loss.to_cpu()
                t1 = time.time()
                dt = t1 - t0
                rate = self.batchsize / dt
                
                
                msgs["E"].append(epoch)
                msgs["L"].append(float(l))

                j += 1
            logs = dict(loss=float(l), epoch=epoch, j=j, prior=float(prior.data), rate=rate)
            print msg.format(**logs)
            print "\n ================================= \n"
            #serializers.save_hdf5("lda2vec.hdf5", self.model)
            msgs["loss_per_epoch"].append(float(l))

        return data, msgs

    # ------------------------------------------------------------------------------------------------------------------
    def initialize_infer(self,
                clambda=200,
                batchsize=4096,
                power=0.75,
                words_pretrained=True,
                temperature=1,
                max_length=1000,
                min_count=0):
        """
        
        :param clambda: 
        :param batchsize: 
        :param power: 
        :param words_pretrained: 
        :param temperature: 
        :param max_length: 
        :param min_count: 
        :return: 
        """
        # 'Strength' of the dircihlet prior; 200.0 seems to work well
        self.clambda = clambda
        # Number of topics to fit
        self.batchsize = batchsize
        # Power for neg sampling
        self.power = power #float(os.getenv('power', 0.75))
        # Intialize with pretrained word vectors
        self.words_pretrained = words_pretrained #bool(int(os.getenv('pretrained', True)))
        self.temp = temperature
        self.max_length = max_length
        self.min_count = min_count

        logging.info('Test parameters initialized!')

    # ------------------------------------------------------------------------------------------------------------------
    def infer(self,docs=None,epochs=200, update_words=False, update_topics=False, topic_vectors=None):
        """
        
        :param docs: 
        :param epochs: 
        :param update_words: 
        :param update_topics: 
        :param topic_vectors: 
        :return: 
        """
        self.preprocess(docs)
        
        logging.info('preprocessed!')
        
        self.infer_model = LDA2Vec(n_documents=self.n_docs,\
                        n_document_topics=self.n_topics,\
                        n_units=300,\
                        n_vocab=self.n_vocab,\
                        counts=self.term_frequency,\
                        n_samples=15,\
                        power=self.power,\
                        temperature=self.temp)
        
        
        if self.words_pretrained:
            self.infer_model.sampler.W.data[:, :] = self.vectors[:self.n_vocab, :]

        self.infer_model.mixture.factors.W.data[:, :] = self.train_model.mixture.factors.W.data
        if topic_vectors is not None:
            assert(topic_vectors.shape==self.infer_model.mixture.factors.W.data.shape), ("topic vectors shape doesn't match")
            self.infer_model.mixture.factors.W.data[:, :] = topic_vectors

        self.infer_model.to_gpu()

        optimizer = O.Adam()
        optimizer.setup(self.infer_model)
        clip = chainer.optimizer.GradientClipping(5.0)
        optimizer.add_hook(clip)
        
        j = 0
        msgs = defaultdict(list)
        for epoch in range(epochs):
            print "epoch : ",epoch
            data = prepare_topics(cuda.to_cpu(self.infer_model.mixture.weights.W.data).copy(),
                                  cuda.to_cpu(self.infer_model.mixture.factors.W.data).copy(),
                                  cuda.to_cpu(self.infer_model.sampler.W.data).copy(),
                                  self.words)
            top_words = print_top_words_per_topic(data)

            if j % 100 == 0 and j > 100:
                coherence = topic_coherence(top_words)
                for j in range(self.n_topics):
                    print j, coherence[(j, 'cv')]

                kw = dict(top_words=top_words, coherence=coherence, epoch=epoch)
                #progress[str(epoch)] = pickle.dumps(kw)
            data['doc_lengths'] = self.doc_lengths
            data['term_frequency'] = self.term_frequency
            #np.savez('topics.pyldavis', **data)

            for d, f in utils.chunks(self.batchsize, self.doc_ids, self.flattened):
                t0 = time.time()
                optimizer.zero_grads()
                l = self.infer_model.fit_partial(d.copy(), f.copy(), update_words=update_words, update_topics=update_topics)
                prior = self.infer_model.prior()
                loss = prior * self.fraction
                loss.backward()
                optimizer.update()
                msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
                       "P:{prior:1.3e} R:{rate:1.3e}")
                prior.to_cpu()
                loss.to_cpu()
                t1 = time.time()
                dt = t1 - t0
                rate = self.batchsize / dt
                
                msgs["E"].append(epoch)
                msgs["L"].append(float(l))
                j += 1

            logs = dict(loss=float(l), epoch=epoch, j=j, prior=float(prior.data), rate=rate)
            print msg.format(**logs)
            print "\n ================================= \n"
            #serializers.save_hdf5("lda2vec.hdf5", self.model)
            msgs["loss_per_epoch"].append(float(l))

        return data, msgs
