# ASGuard: Activation-Scaling Guard to Mitigate Targeted Jailbreaking Attack

## Overview
Large language models (LLMs) often display brittle refusal behaviors that can be bypassed by simple linguistic variations, such as tense jailbreaking. ASGuard (Activation-Scaling Guard) introduces a mechanistically-informed framework to address this vulnerability with precision and interpretability.
(1) Circuit Analysis: Identifies attention heads causally linked to tense-based jailbreaking.
(2) Activation Scaling: Learns channel-wise scaling vectors to recalibrate vulnerable heads.
(3) Preventative Fine-Tuning: Reinforces robust refusal mechanisms while preserving model utility.
Evaluated across multiple LLMs, ASGuard achieves a Pareto-optimal tradeoff between safety and general capabilities, effectively reducing attack success rates while minimizing over-refusal. This work highlights how mechanistic insights can be translated into practical, efficient, and targeted safety interventions, advancing reliable and interpretable AI alignment.

![](assets/overview_1.png)

## Installation
To ensure compatibiliity with other libraries, we recommend using the folliwng versions. You can adjust it based on your environments:

- Python >= 3.10.14
- CUDA 12.2

Follow the order of installation.

1. Clone the repository:
   ```bash
   git clone https://github.com/dmis-lab/ASGuard.git
   cd ASGuard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Implementation
### Step 1: Circuit Analysis
⚠️ Currently we do not support circuit construction. Please wait for an update.

### Step 2: Activation Scaling

```bash
bash run_scaling.sh
```

### Step 3: Preventative Fine-Tuning

```bash
bash run_prevent.sh
```

## Evaluation
⚠️ Currently we do not support evaluation. Please wait for an update.

### Contact
For any questions or issues, feel free to reach out to [522yein (at) korea.ac.kr].