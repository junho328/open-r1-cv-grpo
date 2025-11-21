from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.default_prompts import LETTER_INDICES


def mmlu(line, topic, task_name: str = None):
    query = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    query += "Answer:"

    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    is_few_shots = line.get("__few_shots", False)  # We are adding few shots

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"] if is_few_shots else ["A", "B", "C", "D"],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n",
    )
    
def mmlu_stem(line, task_name: str = None):
    return mmlu(line, "stem", task_name)
    
task = LightevalTaskConfig(
    name="mmlu_stem",
    suite=["community"],
    prompt_function=mmlu_stem,
    hf_repo="TIGER-Lab/MMLU-STEM",
    hf_subset="",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    generation_size=1,
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)

# STORE YOUR EVALS
TASKS_TABLE = [task]
