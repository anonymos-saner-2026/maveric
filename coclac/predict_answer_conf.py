import joblib
from coclac import OpenAILLMClient, CoClaCPipeline

llm = OpenAILLMClient()
coclac = CoClaCPipeline(llm)

# Load lại logistic calibrator đã train
coclac.calibrator = joblib.load("coclac_logistic_calibrator.joblib")
coclac._fitted = True   # đánh dấu đã fitted

q = "Einstein was born in which year?"
a = "Einstein was born in 1879 in Germany."
res = coclac.get_answer_confidence(q, a, agg="min")
print(res)
