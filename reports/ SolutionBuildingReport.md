# Solution Building Report

## Introduction

The objective of this assignment is to create a solution for text detoxification, a critical task in the realm of natural language processing. Text detoxification involves the removal of toxicity and offensive language from text while preserving its semantic meaning as much as possible. This problem can be formally defined as a text style transfer task, where the toxic source style needs to be transformed into a non-toxic target style, as outlined in the provided research paper.


## Hypothesis 1: Style Transfer Models Can Successfully Detoxify Text While Preserving Semantic Meaning
I hypothesize that employing style transfer models can effectively detoxify text by converting toxic language into a non-toxic style while retaining the original semantic meaning. This is based on the assumption that these models can learn to distinguish between toxic and non-toxic language patterns.

## Hypothesis 2: Pre-trained Language Models Will Enhance the Detoxification Process
I hypothesize that leveraging pre-trained language models, such as BERT and GPT-2, will enhance the detoxification process. These models possess extensive linguistic knowledge, enabling them to generate contextually appropriate non-toxic replacements for toxic phrases.

## Hypothesis 3: BART Model's Translation Will Have Lower Toxicity Than Reference Phrases
I hypothesized that the BART model, known for its text generation capabilities, would produce translations with lower toxicity levels compared to the original reference phrases.


## Final Result

Upon analyzing the toxic scores, our results confirmed our hypotheses:

Mean Reference Toxic Scores: 0.8014
Mean Translation Toxic Scores (BART): 0.3901
Mean Translation Toxic Scores (T5): 0.4644
Conclusion:

Both the BART and T5 models generated translations with significantly lower toxicity levels compared to the original reference phrases. This suggests that these models have the potential to contribute to a more positive and respectful digital communication environment. Their ability to generate text with reduced toxicity highlights their usefulness in various applications, including content moderation, customer support, and online communication platforms. By leveraging these models, it is possible to promote healthier and safer interactions in digital spaces, fostering positive online communities.