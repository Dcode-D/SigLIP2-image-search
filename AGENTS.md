# Project guidance

## Goal

Build a research demo for text-to-image retrieval using a pretrained, frozen Google SigLIP 2 checkpoint and the Flickr8k image-caption dataset. The project compares retrieval and reranking methods; its scope is not to modify or fine-tune SigLIP 2.

## Research constraints

- Keep the selected SigLIP 2 checkpoint and its weights unchanged in the primary experiments.
- Make the plain SigLIP 2 text/image embedding cosine search the common baseline.
- Clearly label methods that require a separate pretrained model, learned reranker, training, or generated captions. They are optional comparisons, not changes to SigLIP 2.
- Use a documented Flickr8k split. Prevent test-caption leakage into index construction, prompt/weight selection, or training. Tune any weights only on validation data.
- Define image relevance once and apply it consistently to every method. Report the chosen split and relevance rule with all metrics.
- Keep dataset files, model weights, and generated embeddings out of version control unless explicitly requested.

## Evaluation

At minimum, report text-to-image Recall@1, Recall@5, Recall@10, mean reciprocal rank (MRR), nDCG@10, and query latency. Record the checkpoint, preprocessing, hardware, split, and method parameters so results can be reproduced. Do not claim an improvement from a single anecdotal query; compare on the held-out query set.

## Demo

Provide a Gradio interface when practical. It should accept free-form text, let the user choose among implemented methods, and show ranked Flickr8k images with scores. The demo is illustrative; benchmark metrics come from a separate, reproducible evaluation script.

## Implementation conventions

- Prefer small, understandable components: dataset/index loading, embedding or feature extraction, retrieval methods, evaluation, and UI.
- Cache image embeddings and other reusable features so they are not recomputed for each query.
- Keep method configuration explicit and use a shared result format so methods can be compared fairly.
- Document setup, dataset acquisition requirements, model checkpoint selection, and commands to run evaluation and the demo.
- Do not add a dependency or a new model unless a shortlisted method needs it; explain the need and its role.

## Version control

- Never create a Git commit or push changes to a remote repository on the user's behalf.
- Leave all changes uncommitted so the user can review them first.
- The user is responsible for committing and pushing after review.
