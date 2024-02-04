# convert scibert to sentence transformer for use in setfit

import torch
from sentence_transformers import SentenceTransformer, models

word_embedding_model = models.Transformer("allenai/scibert_scivocab_uncased", max_seq_length=512)

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_sqrt_len_tokens=True,
    pooling_mode_mean_tokens=False,
    pooling_mode_max_tokens=False,
)
# add two trainable feedforward networks to model (DAN)

dense_model = models.Dense(
    in_features=word_embedding_model.get_word_embedding_dimension(),
    out_features=768,
    activation_function=torch.nn.Tanh(),
)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

model.save_to_hub("jordyvl/scibert_scivocab_uncased_sentence_transformer")
