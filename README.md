# binary-recommenders
Demo for binary recommenders

While there are plenty of recommenders out there that rely on ratings to make recommendations, little is known on developing recommenders that utilize a binary metric of "liked" or "not liked." and tag-based systems. While the binary systems can be perceived to be less nuanced/ informative they can effective in contexts where the items being recommended have a binary nature. This repo will demo how to build the folllowing types of recommenders

- a baseline model that recommends the most popular items
- a collabortive filtering (based on likes)
- a tag-based filtering system
- a hybrid model

All models (except of the first one), are built based on a simple cosine similarity metrics.

Follow up
- add examples for metrics to be used for evaluation.