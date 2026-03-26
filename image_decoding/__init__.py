"""Image decoding utilities for Allen-style natural image stimuli.

This subpackage contains helper functions to convert raw spike trains into
firing-rate matrices that can be fed into decoding algorithms as well as
simple baseline decoders.
 
Usage example
-------------
>>> from image_decoding.preprocess import summarise_spikes_to_rates
>>> rates, meta = summarise_spikes_to_rates('core_nll_0/output_bo_bio_trained/chunk_00/spikes.h5')
""" 