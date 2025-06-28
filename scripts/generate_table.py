import json  
from lm_eval.utils import make_table  
  
# 加载您的结果文件  
with open('/home/lishiwei/Select-moe/outputs/2025-06-28/21-17-09/evaluation_results.json', 'r', encoding='utf-8') as f:  
    results = json.load(f)  
  
# 生成表格  
print(make_table(results))  
if "groups" in results:  
    print(make_table(results, "groups"))