# Scope

**Question this project answers:** for a narrow classification task, how much
does QLoRA fine-tuning improve on well-prompted base-model performance, and
what does it cost in training effort versus what it saves at inference?

**Task:** banking intent classification (Banking77, 77 intents; 10,003 train /
3,080 test).

**Comparison rule:** same base model, same held-out test set, same metric
(exact-match accuracy). Only the method changes: few-shot/label-list prompting
versus a QLoRA fine-tune.

**Test set discipline:** the test set is not used for any tuning decision. A
validation split carved from train is used for all choices.

**Out of scope:** full fine-tuning, multi-task training, serving/deployment,
other base models, preference tuning (DPO/RLHF).