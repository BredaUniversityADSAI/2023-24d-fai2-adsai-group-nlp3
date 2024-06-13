.. _model_training:

Model Training
================

This process is used for model training. The model used is RoBERTa model with the input number of output classes.
 See `RoBERTa model <https://huggingface.co/docs/transformers/en/model_doc/roberta>`_ for more information.


.. automodule:: model_training
   :members: load_data, get_model, get_tokenizer, tokenize_text_data, encode_labels, create_tf_datasets,
    preprocess_data, train_model, tokenize_sentences, batch_predict_and_decode, predict, evaluate