import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    """
    Self-supervised objective that masks feature values (not feature names).

    The generator keeps the feature/token identifiers intact while zeroing out
    the values of randomly selected tokens. Targets store the original value
    together with a mask indicator so that the loss can focus only on the masked
    positions.
    """

    def __init__(
        self,
        data,
        discrete_features,
        continuous_features,
        priors,
        tokenizer,
        time=None,
        event=None,
        max_features=0,
        max_features_percentile=95,
        target=None,
        batch_size=32,
        shuffle=True,
        return_index=False,
        add_mask=False,
        augment_copies=None,
        training=None,
        mask_fraction=0.2,
    ):

        features = discrete_features + continuous_features
        self.discrete_features = {i: True for i in discrete_features}
        self.continuous_features = {i: True for i in continuous_features}

        self.add_mask = add_mask
        self.mask_fraction = mask_fraction
        self.data = data
        self.X_dict = data[features].T.to_dict()
        self.features = features

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.priors = priors
        self.with_priors = len(priors) > 0
        self.return_index = return_index

        if max_features == 0:
            non_zero = np.sum(data[self.features].fillna(0) > 0, axis=1)
            self.max_features = int(
                np.floor(np.percentile(non_zero, max_features_percentile))
            )
        else:
            self.max_features = max_features

        self.tokenizer = tokenizer

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X_dict) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X_dict))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def is_valid_feature(self, feature_tuple):
        name, value = feature_tuple
        # Discrete features must not be NaN
        if name in self.discrete_features:
            if np.isnan(value):
                return False
            return True

        # Continuous features: skip NaNs
        return not np.isnan(value)

    def adjust_discrete_zero_value(self, feature_tuple):
        name, value = feature_tuple
        if name in self.discrete_features:
            return name, value + 1, 1, name
        return name, value, 1, name

    def __data_generation(self, indexes):
        seq_len = self.max_features + 1  # +1 for <cls>
        XN = np.zeros((self.batch_size, seq_len), dtype=np.float32)
        XF = np.zeros((self.batch_size, seq_len), dtype=np.float32)

        if self.with_priors:
            XP = np.zeros(
                (self.batch_size, seq_len, self.priors.shape[0]), dtype=np.float32
            )

        value_mask = np.zeros((self.batch_size, seq_len), dtype=np.float32)
        y = np.zeros((self.batch_size, seq_len, 2), dtype=np.float32)

        kx = 0
        for ix, ox in enumerate(indexes):
            features = [
                self.adjust_discrete_zero_value(item)
                for item in self.X_dict[ox].items()
                if self.is_valid_feature(item)
            ]

            if len(features) > self.max_features:
                np.random.shuffle(features)
                features = features[: self.max_features]

            np.random.shuffle(features)

            if self.add_mask and len(features) > 0:
                n_mask = max(1, int(len(features) * self.mask_fraction))
                mask_indices = set(
                    np.random.choice(len(features), size=n_mask, replace=False)
                )
            else:
                mask_indices = set()

            cls_token = ["<cls>", 1.0, 0, "<cls>"]
            padding = [["<pad>", 0.0, 0, "<pad>"]] * (self.max_features - len(features))
            ordered = [cls_token] + features + padding

            row_values = []
            row_tokens = []
            row_mask = []
            row_targets = []

            for pos, entry in enumerate(ordered):
                token_name, token_value, _, original_name = entry
                is_feature = token_name not in ("<cls>", "<pad>")
                is_masked = is_feature and (pos - 1 in mask_indices)

                if is_masked:
                    input_value = 0.0
                    mask_flag = 1.0
                    target_value = token_value
                else:
                    input_value = token_value
                    mask_flag = 0.0
                    target_value = token_value

                row_values.append(input_value)
                row_tokens.append(self.tokenizer.encoder[token_name])
                row_mask.append(mask_flag)
                row_targets.append(target_value)

            XN[kx, :] = row_values
            XF[kx, :] = row_tokens
            value_mask[kx, :] = row_mask
            y[kx, :, 0] = row_mask
            y[kx, :, 1] = row_targets

            if self.with_priors:
                for feat_ix, [feat_id, _, _, _] in enumerate(ordered):
                    try:
                        XP[kx, feat_ix, :] = self.priors[feat_id]
                    except Exception:
                        pass

            kx += 1

        if self.with_priors:
            return [XN[:kx], XF[:kx], XP[:kx], value_mask[:kx]], y[:kx]
        return [XN[:kx], XF[:kx], [], value_mask[:kx]], y[:kx]
