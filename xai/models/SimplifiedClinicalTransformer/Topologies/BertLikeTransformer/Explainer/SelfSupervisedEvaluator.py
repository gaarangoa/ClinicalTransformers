import numpy as np 
import logging

logger = logging.getLogger('ClassifierEvaluator')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
chandler = logging.StreamHandler()
chandler.setLevel(logging.DEBUG)
chandler.setFormatter(formatter)
logger.addHandler(chandler)
logger.setLevel(logging.DEBUG)


def split_vector(N, k):
    # Calculate the number of partitions required
    partitions = N // k
    remaining = N % k
    
    result = []
    start = 0
    
    # Split the vector into smaller vectors
    for i in range(partitions):
        # Calculate the end index of the current partition
        end = start + k
        
        # Add the (start, end) index pair to the result
        result.append((start, end))
        
        # Update the start index for the next partition
        start = end
    
    # If there are remaining elements, add an extra partition
    if remaining > 0:
        end = start + remaining
        result.append((start, end))
    
    return result

class Evaluator():
    def __init__(self, model, epoch=None, return_attentions=False, **kwargs):
        '''
        
        '''

        self.epoch = epoch
        self.trainer = model
        self.return_attentions = return_attentions

        
    def predict(self, data, normalize=True, batch_size=10000, iterations=10):

        '''
        Returns:
        --------
        β: [Iterations, Patients, 1]
        Ŵ: [Iterations, Layers, Patients, Heads, Features, Features]
        Ô: [Iterations, Patients, Features, EmbeddingSize]
        
        --------
        
        '''
        
        # We deactivate the masking tokens as we are generating data. 
        self.trainer.add_mask = False 
        dataloader = self.trainer.data_generator(
            data, 
            batch_size=data.shape[0], 
            normalize=normalize, 
            max_features=self.trainer.max_features, 
            shuffle=False,
            augment_copies=1
        )
        
        predictions = []
        attentions = []
        outputs = []
        tokens = []
        risks = []
        labels = []
        for iteration in range(iterations):
        
            X, y = dataloader.__getitem__(0)
            labels.append(y)

            Ŷ = []
            Ŵ = []
            Ô = []
            R = []
            
            output = []
            counts = 0
            total = X[0].shape[0]
            indexes = split_vector(X[0].shape[0], k=batch_size)

            for start, end in indexes:

                x = [ r[start:end] for r in X[:3] ]
                # token predictions / value predictions / attention weights / embeddings
                ỹ, y2, ŵ, ô = self.trainer.model(x, training=False)

                Ŷ.append(ỹ.numpy())
                if self.return_attentions == True:
                    Ŵ.append(np.array([i.numpy() for i in ŵ]))
                Ô.append(ô.numpy())
                
                counts += batch_size

            Ŷ = np.concatenate(Ŷ)
            if self.return_attentions:
                Ŵ = np.concatenate(Ŵ)
            Ô = np.concatenate(Ô)
            
            features = np.array([[self.trainer.tokenizer.decoder[int(i)] for i in p] for p in X[1]])
            
            tokens.append(features)
            predictions.append(Ŷ)
            attentions.append(Ŵ)
            outputs.append(Ô)

        return np.array(predictions), np.array(attentions), np.array(outputs), np.array(tokens), np.array(labels)


    def embeddings(self, data, sample_id, o, t):
        '''
        Parse output from the transformer and get the outputs, features, 
        patient ids and iterations as a giant table. This is useful for downstreatm tasks
            data: dataset, 
            sample_id: sample_id
            o: embeddings
            t: tokens
        '''
        patients = data[sample_id]

        outputs = []
        features = []
        patient_ids = []
        iterations = []
        risks = []
        for iteration in range(o.shape[0]):
            for patient_ix, patient in enumerate(patients):

                ô = o[iteration, patient_ix, :, :]
                t̂ = t[iteration, patient_ix, :]

                no_pad_out = ô[ (t̂ != '<pad>') & (t̂ != '<mask>')]
                no_pad_features = t̂[ (t̂ != '<pad>') & (t̂ != '<mask>')]

                outputs.append(no_pad_out)
                features.append(no_pad_features)
                patient_ids.append([patient]*len(no_pad_features))
                iterations.append([iteration]*len(no_pad_features))

        outputs = np.concatenate(outputs, axis=0)
        features = np.concatenate(features)
        patient_ids = np.concatenate(patient_ids)
        iterations = np.concatenate(iterations)
        
        return outputs, features, patient_ids, iterations