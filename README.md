# Decoding Parameter and Sampling Analysis

Group SSG compared Llama 3.2:3B and Qwen 2.5:3B locally through Ollama. One fixed prompt was tested under 64 settings with three seeds per model (384 responses).

| Member | Registration number |
| --- | --- |
| Anusha Baig | 2026-MS-AI-2026 |
| Saira Alamgir | 2026-MS-AI-03 |
| Waqad Qayyum Ch | 2026-MS-AI-3100 |

## Files
- Project_Report.pdf: methods, results, graphs and limitations.
- Decoding_Parameter_Analysis.ipynb: submitted notebook with saved outputs.
- decoding_analysis.py: script using the notebook's experiment and scoring logic.
- decoding_results.csv: original model responses.
- scored_results.csv: response scores.
- parameter_summary.csv: scores for each model/settings combination.
- *_comparison.png: comparison graphs.

## Setup
Install Python and Ollama, then run from this folder:
```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```
Keep Ollama running for generation.

## Reproduce analysis using the supplied results
```bash
python decoding_analysis.py --analyze-only
```
This recalculates scores and graphs without requesting model responses. The script writes outputs in its own folder.

## Generate responses
```bash
python decoding_analysis.py
```
Existing experiments in decoding_results.csv are skipped. For a completely new run, use a separate copy of the project folder and rename its decoding_results.csv before running. Do not combine results from changed prompts or model versions.

## Open the notebook
```bash
python -m notebook
```
Open Decoding_Parameter_Analysis.ipynb and run cells in order. The supplied notebook already includes outputs.

## Findings and scope
Both models achieved 100% on the three short factual answers and the JSON schema check. Overall pairwise exact agreement was 45.83% for Llama and 33.33% for Qwen. Both reached 100% agreement at temperature 0.0. Agreement measures identical text, not semantic quality.

Factual accuracy was evaluated only for the three short-answer questions; the seasons explanation was not assessed. These results are not a general hallucination benchmark. Exact environment versions were not recorded in the submitted experiment, and requirements.txt is not a frozen environment.
